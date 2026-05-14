# DiT S3-Cache: c_FID (FID Cache Sensitivity) 實作 Prompt

## 目標

實作 DiT-XL/2 256×256 的 FID Cache Sensitivity 實驗，產出每個 DiTBlock 在不同 cache frequency `k` 下的 FID 敏感度 `delta_FID`。這是 S3-Cache Stage 0 的第三條 evidence，Stage 1 需要它來計算 cost function $J$ 中的 per-block weight $w_b^{\text{clip}}$。

## 核心邏輯（一句話版本）

對每個 block（28 個），單獨對該 block 啟用 cache（每 k 步才重算），其他 27 個 block 正常計算，然後生成圖片計算 FID，與 baseline FID 做差得到 `delta_FID`。

## 參考實作

Diff-AE 版本已完成，位於：
```
QATcode/cache_method/c_FID/fid_cache_sensitivity/fid_cache_sensitivity.py
QATcode/cache_method/c_FID/fid_cache_sensitivity/fid_sensitivity_results.json
QATcode/cache_method/c_FID/fid_cache_sensitivity/run_experiment.sh
```

**務必先 trace 上述實作**，理解以下關鍵設計：
- `create_simple_cache_config()` 如何生成 per-layer cache scheduler
- `evaluate_fid_with_cache()` 如何將 cache scheduler 注入 sampling
- `should_skip_experiment()` 的增量續跑機制
- JSON 結構（`baseline_fid`, `k3/k4/k5`, per-layer `fid` + `delta`）

## DiT 與 Diff-AE 的關鍵差異

### 簡化的部分（DiT 比 Diff-AE 簡單很多）

1. **不需要量化**：DiT 是 FP32 pretrained model，直接載入即可。不需要 `QuantModel`、`LoRA`、`calibration data`、`uint8 weight conversion` 等所有量化相關程式碼。模型載入就是：
```python
model = DiT_models["DiT-XL/2"](input_size=32, num_classes=1000).to(device)
model.load_state_dict(find_model("DiT-XL-2-256x256.pt"))
model.eval()
```

2. **Block 命名更簡單**：28 個 homogeneous DiTBlock，用 `block_0` ~ `block_27` 即可。不需要 Diff-AE 的 `encoder_layer_X` / `middle_layer` / `decoder_layer_X` 三段式命名。

3. **不需要 EMA model 同步**：DiT 的 pretrained checkpoint 就是最終 weights，沒有 EMA 分離。

### 需要處理的新問題

1. **CFG（Classifier-Free Guidance）**：
   - DiT sampling 時 cond + uncond 拼成 2B batch 一起跑 forward
   - Cache 機制需要在這個 batched forward 中對指定 block 使用前一步的 residual
   - CFG 合成發生在 `forward_with_cfg` 中，在所有 block forward 之後
   - **Cache 時 cond 和 uncond 共用同一個 schedule**（已決定）

2. **DDPM 250 steps**（vs Diff-AE 的 DDIM 20/100 steps）：
   - 每次 FID evaluation 的 sampling 時間更長
   - k 值的選擇可能需要不同（Diff-AE 用 k∈{3,4,5}，DiT 250 步的 k 可以更大）

3. **FID 計算流程**：
   - 使用 `sample_ddp.py` 的模式：生成 .png → 合併 .npz → ADM evaluator
   - 需要 ImageNet 256×256 的 reference statistics
   - 建議 FID@1K 做 sensitivity sweep，最終驗證用 FID@50K

## 實作規格

### 檔案結構

```
dit_s3cache/
├── evidence/
│   ├── collect_evidence.py    ← 已完成（Evidence 1+2）
│   ├── hooks.py               ← 已完成
│   └── utils.py               ← 已完成
├── cfid/
│   ├── fid_cache_sensitivity.py   ← 新增：主腳本
│   ├── cache_runner.py            ← 新增：cache-aware sampling engine
│   └── fid_sensitivity_results.json  ← 自動產生：結果
└── scripts/
    └── run_cfid.sh                ← 新增：啟動腳本
```

### Task 1: Cache-Aware Sampling Engine (`cache_runner.py`)

這是最核心的部分。需要修改 DiT 的 sampling 流程，讓指定 block 在非 recompute timestep 使用前一步的 residual。

**設計要點**：

1. **Cache scheduler 格式**：
```python
# cache_scheduler: dict[int, set[int]]
# key = block index (0~27)
# value = set of timesteps where this block should RECOMPUTE
# 不在 set 裡的 timestep → 使用前一步的 cached residual
# 
# 例如：block_5 每 3 步重算一次，250 步：
# cache_scheduler[5] = {249, 246, 243, 240, ...}  (sampling order, 從大到小)
# 其他 block：set(range(250))  (每步都重算)
```

2. **CachedDiTBlock**：改寫 DiTBlock.forward，加入 cache 邏輯

```python
class CachedDiTBlock:
    """Wraps a DiTBlock with optional per-timestep caching."""

    def __init__(self, block, block_idx, cache_scheduler, total_steps):
        self.block = block
        self.block_idx = block_idx
        self.recompute_steps = cache_scheduler.get(block_idx, set(range(total_steps)))
        self.cached_residual = None  # 前一步的 r_msa + r_mlp
        self.current_step = None

    def forward(self, x, c):
        if self.current_step in self.recompute_steps:
            # 正常計算
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                self.block.adaLN_modulation(c).chunk(6, dim=1)
            r_msa = gate_msa.unsqueeze(1) * self.block.attn(
                modulate(self.block.norm1(x), shift_msa, scale_msa))
            x_after_msa = x + r_msa
            r_mlp = gate_mlp.unsqueeze(1) * self.block.mlp(
                modulate(self.block.norm2(x_after_msa), shift_mlp, scale_mlp))
            x = x_after_msa + r_mlp

            # 存 cache
            self.cached_residual = (r_msa + r_mlp).detach()
        else:
            # 使用 cache
            assert self.cached_residual is not None, \
                f"Block {self.block_idx} step {self.current_step}: cache miss"
            x = x + self.cached_residual

        return x
```

**⚠️ 關鍵注意事項**：
- 第一個 sampling step（timestep=249）的所有 block 都**必須** recompute（沒有前一步可以 cache）
- `current_step` 需要在每個 diffusion step 開始前由外部設定
- `cached_residual` 的 shape 是 `(2B, 256, 1152)`，包含 cond + uncond

3. **Sampling loop 改寫**：

不能直接用 `diffusion.p_sample_loop`，因為需要在每個 timestep 前更新 `current_step`。需要用 `p_sample_loop_progressive` 或手寫 loop：

```python
def sample_with_cache(model, diffusion, z, model_kwargs, cache_scheduler, n_steps, device):
    """Run DDPM sampling with block-level caching."""

    # 1. Install cache wrappers
    cached_blocks = install_cache_wrappers(model, cache_scheduler, n_steps)

    # 2. Create cached forward_with_cfg
    # 需要一個 wrapper 讓 diffusion.p_sample 呼叫時能更新 current_step
    # ...

    # 3. Run sampling
    indices = list(range(n_steps))[::-1]  # [249, 248, ..., 0]
    img = z.clone()
    for step_idx, i in enumerate(indices):
        # 更新所有 cached block 的 current_step
        for cb in cached_blocks:
            cb.current_step = i  # 用 diffusion timestep, 不是 step_idx
            # ← 這裡要跟 cache_scheduler 裡 recompute_steps 的 convention 一致！

        t = torch.tensor([i] * z.shape[0], device=device)
        with torch.no_grad():
            out = diffusion.p_sample(
                model.forward_with_cfg,  # 或 cached 版本
                img, t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )
        img = out["sample"]

    # 4. Restore original forwards
    restore_cache_wrappers(cached_blocks)

    return img
```

**Timestep convention 務必統一**：
- `p_sample_loop_progressive` 的 indices 是 `[249, 248, ..., 0]`
- `cache_scheduler` 中 recompute_steps 用的是 diffusion timestep（即 `i` 的值）
- 第一個執行的 timestep 是 249，必須在所有 block 的 recompute set 中

### Task 2: Cache Config 生成

```python
def create_dit_cache_config(
    target_block: int,
    k: int,
    total_steps: int = 250,
    n_blocks: int = 28,
) -> dict[int, set[int]]:
    """
    為單一 block 生成 cache 配置。

    Args:
        target_block: 要測試 cache sensitivity 的 block index (0~27)
        k: cache frequency（每 k 個 step 重算一次）
        total_steps: DDPM 總步數 (250)
        n_blocks: 總 block 數 (28)

    Returns:
        dict: {block_idx: set of timesteps to recompute}
    """
    indices = list(range(total_steps))[::-1]  # [249, 248, ..., 0]

    # Target block: 每 k 步重算一次
    recompute_steps = set()
    for step_idx, timestep in enumerate(indices):
        if step_idx % k == 0:
            recompute_steps.add(timestep)
    # 確保第一步一定重算
    recompute_steps.add(indices[0])  # timestep 249

    # 所有 block 的 scheduler
    scheduler = {}
    for b in range(n_blocks):
        if b == target_block:
            scheduler[b] = recompute_steps
        else:
            scheduler[b] = set(range(total_steps))  # 每步都算
    return scheduler
```

### Task 3: 主腳本 (`fid_cache_sensitivity.py`)

**整體流程**：

```python
def main():
    # 1. 載入 DiT 模型（簡單！）
    model = load_dit_model(device)

    # 2. 載入或建立 results JSON
    results = load_results(results_path)

    # 3. 跑 baseline（如果還沒跑）
    if "baseline_fid" not in results:
        baseline_fid = generate_and_compute_fid(
            model, diffusion, cache_scheduler=None, ...)
        results["baseline_fid"] = baseline_fid
        save_results(results)

    # 4. 對每個 (k, block) 組合跑 cache sensitivity
    for k in k_values:
        for block_idx in range(28):
            if should_skip(results, k, block_idx):
                continue

            scheduler = create_dit_cache_config(block_idx, k, total_steps=250)
            fid = generate_and_compute_fid(
                model, diffusion, cache_scheduler=scheduler, ...)
            delta = fid - baseline_fid

            results[f"k{k}"][f"block_{block_idx}"] = {"fid": fid, "delta": delta}
            save_results(results)  # 增量存檔
```

### Task 4: FID 計算

FID 計算需要：
1. 用 cache-aware sampling 生成 N 張圖（N=1000 for FID@1K）
2. VAE decode latent → pixel images
3. 存為 .png 或合併成 .npz
4. 用 ADM evaluator 計算 FID（需要 ImageNet 256×256 reference stats）

**建議直接復用 `sample_ddp.py` 的邏輯**，但替換 sampling 函數為 cache-aware 版本。

```python
def generate_and_compute_fid(
    model, diffusion, vae, cache_scheduler,
    num_samples=1000, batch_size=8, cfg_scale=4.0,
    ref_stats_path="VIRTUAL_imagenet256_labeled.npz",
    device="cuda",
):
    """Generate samples with optional caching and compute FID."""

    # 1. Generate latents
    all_samples = []
    n_iterations = math.ceil(num_samples / batch_size)

    for i in range(n_iterations):
        z, y = make_cfg_inputs(batch_size, latent_size=32, num_classes=1000, device=device)
        model_kwargs = {"y": y, "cfg_scale": cfg_scale}

        if cache_scheduler is not None:
            samples = sample_with_cache(model, diffusion, z, model_kwargs,
                                        cache_scheduler, n_steps=250, device=device)
        else:
            samples = diffusion.p_sample_loop(
                model.forward_with_cfg, z.shape, z,
                clip_denoised=False, model_kwargs=model_kwargs,
                device=device, progress=False)

        # CFG: 取前 B
        samples, _ = samples.chunk(2, dim=0)

        # VAE decode
        samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255)
        samples = samples.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        all_samples.append(samples)

    all_samples = np.concatenate(all_samples)[:num_samples]

    # 2. Compute FID
    fid = compute_fid_from_samples(all_samples, ref_stats_path)
    return fid
```

**FID 計算選項**：
- **選項 A**（推薦）：用 `pytorch-fid` 或 `clean-fid` 套件，直接從 numpy array 算
- **選項 B**：存成 .npz，用 ADM evaluator `evaluator.py` 算（跟 DiT 論文一致）
- 兩者都行，選項 A 更方便，選項 B 跟論文更一致。先用 A 快速迭代，最終結果用 B 驗證。

### Task 5: 雙主機分割執行

兩台主機各一張 RTX 5090。DiT sampling 是 compute-bound，PCIe Gen 3 vs Gen 4 對 throughput 影響 < 2%，視為等速，直接均分。

#### 5a: Task Dispatch (`get_task_list`)

```python
def get_task_list(part: str | None) -> list[tuple[int, int]]:
    """Return (k, block_idx) pairs for the given part.

    Partitioning (42 + 42 = 84 cached evals, balanced):
      Part A: k=3 全部 (28) + k=5 blocks 0~13 (14)  = 42
      Part B: k=5 blocks 14~27 (14) + k=10 全部 (28) = 42

    Each part also runs its own baseline (for cross-validation).
    """
    if part == "A":
        tasks = [(3, b) for b in range(28)]
        tasks += [(5, b) for b in range(14)]
    elif part == "B":
        tasks = [(5, b) for b in range(14, 28)]
        tasks += [(10, b) for b in range(28)]
    else:
        # 單機全跑
        tasks = []
        for k in [3, 5, 10]:
            tasks += [(k, b) for b in range(28)]
    return tasks
```

**分割理由**：
- k=3 整包給 A，k=10 整包給 B，k=5 中間切半
- 每個 k 至少有一邊是完整的，方便中途看 partial results 判斷趨勢
- 兩邊各自跑 baseline，合併時可交叉驗證（差異應 < 0.5）

#### 5b: CLI 設計

```python
# 在 argparse 中加入
parser.add_argument(
    "--part",
    type=str,
    default=None,
    choices=["A", "B"],
    help=(
        "雙主機分片: "
        "A = baseline + k=3 全部 + k=5 blocks 0~13 (共 43 evals); "
        "B = baseline + k=5 blocks 14~27 + k=10 全部 (共 43 evals). "
        "不指定則單機全跑 (85 evals)."
    ),
)
```

#### 5c: main() 整合

```python
def main(args):
    model, diffusion, vae = load_everything(args)
    results = load_results(results_path)

    # 每個 part 都先跑自己的 baseline
    baseline_fid = ensure_baseline_fid(model, diffusion, vae, results, results_path)

    # 取得該 part 的 task list
    tasks = get_task_list(args.part)
    print(f"Part {args.part or 'FULL'}: {len(tasks)} cached evals + 1 baseline")

    for i, (k, block_idx) in enumerate(tasks, 1):
        if should_skip(results, k, block_idx):
            print(f"[{i}/{len(tasks)}] k={k} block_{block_idx} — 已完成，跳過")
            continue

        print(f"[{i}/{len(tasks)}] k={k} block_{block_idx}")
        scheduler = create_dit_cache_config(block_idx, k, total_steps=250)
        fid = generate_and_compute_fid(model, diffusion, vae, scheduler, ...)
        delta = fid - baseline_fid

        k_key = f"k{k}"
        if k_key not in results["results"]:
            results["results"][k_key] = {}
        results["results"][k_key][f"block_{block_idx}"] = {"fid": fid, "delta": delta}
        save_results(results, results_path)
```

#### 5d: 啟動腳本 (`run_cfid.sh`)

```bash
#!/bin/bash
# DiT S3-Cache c_FID sensitivity sweep — dual-host version
#
# 主機 1:  bash run_cfid.sh A
# 主機 2:  bash run_cfid.sh B
# 單機全跑: bash run_cfid.sh
set -e

SCRIPT="dit_s3cache/cfid/fid_cache_sensitivity.py"
PART="${1:-}"

if [[ -n "$PART" ]]; then
    if [[ "$PART" != "A" && "$PART" != "B" ]]; then
        echo "Usage: $0 [A|B]"
        echo "  A = baseline + k=3 all + k=5 blocks 0~13  (43 evals)"
        echo "  B = baseline + k=5 blocks 14~27 + k=10 all (43 evals)"
        echo "  (no arg) = single-host full sweep (85 evals)"
        exit 1
    fi
    echo "=========================================="
    echo "c_FID Sensitivity — Part ${PART}"
    echo "=========================================="
    python3 "$SCRIPT" --part "$PART" --num-fid-samples 1000 \
        2>&1 | tee "dit_s3cache/cfid/cfid_part_${PART}.log"
else
    echo "=========================================="
    echo "c_FID Sensitivity — Full (single host)"
    echo "=========================================="
    python3 "$SCRIPT" --num-fid-samples 1000 \
        2>&1 | tee "dit_s3cache/cfid/cfid_full.log"
fi

echo "Done!"
```

#### 5e: 合併兩台結果

兩台跑完後各自有一份 `fid_sensitivity_results.json`。合併方式：

```python
def merge_results(path_a: str, path_b: str, output: str):
    """Merge Part A and Part B JSON results."""
    a = json.load(open(path_a))
    b = json.load(open(path_b))

    merged = a.copy()

    # Cross-validate baselines
    bl_a = a["results"]["baseline_fid"]
    bl_b = b["results"]["baseline_fid"]
    assert abs(bl_a - bl_b) < 0.5, f"Baseline mismatch: A={bl_a:.4f} B={bl_b:.4f}"
    merged["results"]["baseline_fid"] = (bl_a + bl_b) / 2  # 取平均

    # Merge k entries
    for key in b["results"]:
        if key == "baseline_fid":
            continue
        if key not in merged["results"]:
            merged["results"][key] = {}
        merged["results"][key].update(b["results"][key])

    json.dump(merged, open(output, "w"), indent=2)
    print(f"Merged → {output} (baseline avg: {merged['results']['baseline_fid']:.4f})")
```

可以做成獨立小腳本 `merge_cfid_results.py`，或直接放在 `fid_cache_sensitivity.py` 裡加一個 `--merge` flag。

## 超參數建議

| Parameter | Value | Notes |
|-----------|-------|-------|
| k_values | {3, 5, 10} | 250 步比 Diff-AE 的 20/100 步多很多，k 可以更大 |
| num_fid_samples | 1000 | FID@1K for sweep，最終驗證用 50K |
| batch_size (per-side) | 8 | 2B=16 進 model。需載入 VAE，memory 比 evidence collection 更吃緊 |
| cfg_scale | 4.0 | 與 evidence collection 和 sample.py 一致 |
| num_sampling_steps | 250 | DDPM full steps |
| seed | 0 | FID evaluation 用固定 seed 確保 baseline 和 cached 生成同分佈（但不需要同一組圖） |

### k 值選擇說明

Diff-AE 用 k∈{3,4,5} 是因為只有 20/100 步，k=5 在 20 步中已經很激進（只算 4 次）。

DiT 有 250 步，k∈{3,4,5} 在 250 步中太保守（k=5 仍然算 50 次/block）。建議：
- **k=3**：溫和，baseline 對照
- **k=5**：中等
- **k=10**：激進，每 10 步重算一次（25 次/block），更接近實際 cache 情境

也可以視第一輪結果追加 k=15 或 k=20。

## JSON 輸出格式

維持跟 Diff-AE 相同的結構，方便比較：

```json
{
  "config": {
    "model": "DiT-XL/2",
    "image_size": 256,
    "num_sampling_steps": 250,
    "cfg_scale": 4.0,
    "eval_samples": 1000,
    "fid_method": "pytorch-fid or ADM"
  },
  "results": {
    "baseline_fid": 9.62,
    "k3": {
      "block_0": {"fid": 10.15, "delta": 0.53},
      "block_1": {"fid": 9.70, "delta": 0.08},
      ...
      "block_27": {"fid": 12.34, "delta": 2.72}
    },
    "k5": { ... },
    "k10": { ... }
  },
  "last_updated": "2026-05-14 ..."
}
```

## 驗證檢查清單

- [ ] **Baseline 正確性**：不使用 cache 時，FID 與官方 DiT-XL/2 256×256 reported 數字大致吻合（FID@50K ~9.62；FID@1K 會偏高但應在合理範圍）
- [ ] **Cache 不改變其他 block**：當 target_block=5 時，其他 27 個 block 的計算與 baseline 完全一致
- [ ] **第一步一定 recompute**：所有 block 在 timestep=249 都有重算，不會出現 cache miss
- [ ] **增量續跑**：中斷後重跑，已完成的 (k, block) 組合會跳過
- [ ] **Memory 穩定**：cached_residual 只存一份（上一步的），不會隨 step 累積
- [ ] **FID 可重現**：同一 config 跑兩次，FID 差異 < 0.5（random seed 控制）

## 成本估算

- 總共 85 次 FID@1K evaluation（1 baseline + 3k × 28 blocks）
- 每次 evaluation：~125 iterations × 250 steps ≈ 31,250 forward passes
- **單 GPU（5090）預計 30~40 小時**
- **雙主機分割後：每台 43 evals ≈ 15~20 小時**，兩台同時結束

| | Part A | Part B |
|---|---|---|
| Baseline | 1 | 1 |
| k=3 blocks 0~27 | 28 | — |
| k=5 blocks 0~13 | 14 | — |
| k=5 blocks 14~27 | — | 14 |
| k=10 blocks 0~27 | — | 28 |
| **Total evals** | **43** | **43** |

## 與已完成的 Evidence 1+2 的關係

c_FID 與 `collect_evidence.py`（similarity + SVD）完全獨立：
- 不共用 hook（c_FID 用 CachedDiTBlock，evidence 用 recording hook）
- 不共用 sampling run
- 可以在不同 GPU 上平行跑
- 三條 evidence 全部完成後，進入 Stage 1 scheduler synthesis

> 請補充：
> 1. FID 的計算可以參考 DiT/README.md 中的Evaluation。
> 2. 合併的部分我會自己後續手動進行合併。
> 3. 我期望在進行 c_FID 的實驗的時候，實驗中生成的圖片我自己是建議放在 dit_s3cache/fid/gen_image/ 底下；實驗程式碼放在 dit_s3cache/fid 底下；每次進行完FID實驗後，清空 gen_image 底下內容。
