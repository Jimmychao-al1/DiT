# DiT S3-Cache: Evidence Collection 實作 Prompt

## 專案目標

將 S3-Cache 的 Stage 0 (Evidence Collection) 移植到 DiT-XL/2 256×256 (ImageNet class-conditional)。
需要在一次 sampling run 中同時收集兩條 evidence：Cross-Timestep Similarity 和 SVD Subspace Drift。

## 架構背景

### S3-Cache 概述
S3-Cache 是一個三階段 offline pipeline，用於加速 diffusion model inference：
- **Stage 0 (Evidence Collection)**：收集每個 block 在每個 timestep 的 residual output 統計量 ← 你現在要做這個
- Stage 1：基於 evidence 合成 cache scheduler
- Stage 2：refinement

核心思路：如果某個 block 在連續 timestep 間的 residual output 變化很小，就可以 cache 住前一個 timestep 的結果，跳過該 block 的計算。

### DiT-XL/2 架構要點（已 trace 確認）

1. **28 個 DiTBlock**，flat sequential（無 skip connections）
2. **每個 DiTBlock 的 forward**：
```python
def forward(self, x, c):
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
        self.adaLN_modulation(c).chunk(6, dim=1)
    # Cache target 1: MHSA residual branch
    r_msa = gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
    x = x + r_msa
    # Cache target 2: MLP residual branch
    r_mlp = gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
    x = x + r_mlp
    return x
```
3. **Block-level caching**：cache `block_residual = r_msa + r_mlp`，shape `(2B, T_tokens=256, D=1152)`
4. **CFG 是 batched**：cond + uncond 拼成 2B 一起跑 forward。`forward_with_cfg` 在 forward 之後才做 CFG 合成。
5. **Conditioning**：`c = t_embed + y_embed`，在 block loop 外計算一次，所有 block 共用同一個 `c`
6. **Sampler**：DDPM 250 steps, linear beta schedule (β_start=0.0004, β_end=0.08)
7. **Model output**：ε + learned variance（`learn_sigma=True`，output 8 channels, split 成 ε 和 variance）

### CFG 處理策略（已決定）
- Evidence 收集時 **不區分 cond/uncond path**，直接在 2B batch 上整體計算
- Cache schedule 是 `(block, timestep)` 的 binary mask，cond/uncond 共用同一個 schedule
- 理由：它們在同一個 batched forward 裡，拆開處理複雜度大增且收益不明

## 既有實作參考

> **[浤銘補充區]** 請在這裡提供 Diff-AE / LDM S3-Cache evidence collection 的相關檔案路徑，讓 Cursor 可以 trace：
>
> ```
> # Diff-AE S3-Cache evidence collection:
> /home/jimmy/diffae/QATcode/cache_method
> cache_method/a_L1_L2_cosine
> cache_method/b_SVD
> cache_method/c_FID
> cache_method/Stage0
>
> # LDM S3-Cache evidence collection:
> /home/jimmy/latent-diffusion/ldm_S3cache
> cache_method/a_L1_L2_cosine
> cache_method/b_SVD
> cache_method/c_FID
> cache_method/Stage0
>
> # 特別注意這些檔案中的：
> # - hook 註冊方式（register_forward_hook vs 改寫 forward）
> # - evidence 計算的具體函數
> # - 輸出的 .npz 格式
> # - memory 管理策略（是否逐 timestep 存、batch 累積等）
> ```

> 如果有任何設計上的疑問，可提出。

## 實作規格

### 檔案結構
```
dit_s3cache/
├── evidence/
│   ├── collect_evidence.py    ← 主腳本，一次 run 收集 similarity + svd
│   ├── hooks.py               ← DiTBlock hook 機制
│   └── utils.py               ← evidence 計算函數
├── configs/
│   └── evidence_config.yaml   ← 可選，超參數集中管理
└── scripts/
    └── run_evidence.sh        ← 啟動腳本
```

### Task 1: Hook 機制 (`hooks.py`)

**目標**：攔截每個 DiTBlock 的 residual outputs，不改動原始 `models.py`。

**建議方式**：用 wrapper class 或 monkey-patch `DiTBlock.forward`，在每次 forward 時把 `r_msa + r_mlp` 存入一個外部 buffer。

```python
class DiTBlockHook:
    """
    攔截 DiTBlock 的 residual output。
    
    用法：
        hooks = [DiTBlockHook(block, block_idx) for block_idx, block in enumerate(model.blocks)]
        # ... 跑 sampling ...
        residuals = [h.get_residual() for h in hooks]  # list of (2B, 256, 1152)
        [h.clear() for h in hooks]
    """
```

**關鍵設計決策**：
- 改寫 `DiTBlock.forward` 來同時計算並存儲 `r_msa + r_mlp`（原始 code 中這兩個是 inline 的，沒有命名變數）
- 每個 timestep 結束後，立即計算 evidence 並釋放 residual tensor，**不要**把所有 250 steps 的 residual 都存在 memory 中
  - 28 blocks × (2×8, 256, 1152) × fp32 ≈ 每 timestep ~2.7 GB，250 steps 全存 = ~675 GB，不可行
  - **正確做法**：只保留「前一個 timestep 的 residual」用於計算 similarity/svd drift，計算完立即丟棄

**Monkey-patch 範例**：
```python
def patch_dit_block(block, block_idx, storage):
    original_forward = block.forward
    
    def hooked_forward(x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            block.adaLN_modulation(c).chunk(6, dim=1)
        r_msa = gate_msa.unsqueeze(1) * block.attn(modulate(block.norm1(x), shift_msa, scale_msa))
        x = x + r_msa
        r_mlp = gate_mlp.unsqueeze(1) * block.mlp(modulate(block.norm2(x), shift_mlp, scale_mlp))
        x = x + r_mlp
        
        # Store block residual
        storage[block_idx] = (r_msa + r_mlp).detach()  # (2B, 256, 1152)
        return x
    
    block.forward = hooked_forward
```

注意：`modulate` 函數在 `models.py` 中是 module-level function，hook 時需要 import 或直接 reference。

### Task 2: Evidence 計算函數 (`utils.py`)

#### Evidence 1: Cross-Timestep Similarity

```python
def compute_similarity(
    residual_current: torch.Tensor,  # (2B, 256, 1152) — 當前 timestep
    residual_prev: torch.Tensor,     # (2B, 256, 1152) — 前一個 timestep
) -> dict:
    """
    Returns:
        l1_diff: scalar (averaged over batch)
        cos_sim: scalar (averaged over batch)
    """
    # Flatten tokens+channels: (2B, 256*1152)
    r_curr = residual_current.flatten(1)
    r_prev = residual_prev.flatten(1)
    
    # L1 relative difference
    l1_diff = (r_curr - r_prev).abs().sum(dim=1) / r_curr.abs().sum(dim=1).clamp(min=1e-8)
    
    # Cosine similarity
    cos_sim = F.cosine_similarity(r_curr, r_prev, dim=1)
    
    return {
        "l1_diff": l1_diff.mean().item(),
        "cos_sim": cos_sim.mean().item(),
    }
```

#### Evidence 2: SVD Subspace Drift

```python
def compute_svd_drift(
    residual_current: torch.Tensor,  # (2B, 256, 1152)
    V_prev: torch.Tensor or None,    # (k, 1152) — 前一個 timestep 的 top-k 右 singular vectors
    k: int = 16,
) -> dict:
    """
    Returns:
        grassmann_dist: scalar
        subspace_overlap: scalar
        sv_spectrum: np.array shape (k,)
        V_current: (k, 1152) — 供下一個 timestep 使用
    """
    # Reshape to 2D matrix: (2B*256, 1152)
    R = residual_current.reshape(-1, 1152).float()  # 確保 fp32
    R_centered = R - R.mean(dim=0, keepdim=True)
    
    # Truncated SVD
    # 注意：torch.linalg.svd 在大矩陣上很慢，
    # 考慮用 torch.svd_lowrank 或先 subsample tokens
    U, S, Vh = torch.svd_lowrank(R_centered, q=k)  # 更快的 randomized SVD
    V_current = Vh.T  # (1152, k) → 取 .T 看 convention，確保 V_current shape 是 (k, 1152)
    # 注意：torch.svd_lowrank 返回 (U, S, V) where V shape is (1152, k)
    # 所以 V_current = V.T → (k, 1152)
    V_current = Vh.T.T  # ... 請仔細確認 torch.svd_lowrank 的返回值 convention
    
    sv_spectrum = S[:k].cpu().numpy()
    
    if V_prev is None:
        return {
            "grassmann_dist": float('nan'),
            "subspace_overlap": float('nan'),
            "sv_spectrum": sv_spectrum,
            "V_current": V_current.detach(),
        }
    
    # Grassmann distance
    cross = V_current @ V_prev.T  # (k, k)
    sigma = torch.linalg.svdvals(cross)
    angles = torch.acos(sigma.clamp(-1 + 1e-7, 1 - 1e-7))
    
    return {
        "grassmann_dist": angles.norm().item(),
        "subspace_overlap": sigma.sum().item() / k,
        "sv_spectrum": sv_spectrum,
        "V_current": V_current.detach(),
    }
```

**⚠️ `torch.svd_lowrank` 的返回值**：請在實作時先跑一個小 test 確認 `(U, S, V)` 的 shape convention。文件說返回 `V` shape `(n, k)`，那麼 top-k 右 singular vectors 就是 `V.T`，shape `(k, n)`。

### Task 3: 主腳本 (`collect_evidence.py`)

**整體流程**：

```python
def main():
    # 1. Load model (same as sample.py)
    model = DiT_XL_2(input_size=32, num_classes=1000).to(device)
    state_dict = find_model("DiT-XL-2-256x256.pt")
    model.load_state_dict(state_dict)
    model.eval()
    
    diffusion = create_diffusion("250")
    
    # 2. Install hooks
    storage = {}  # block_idx → residual tensor
    for idx, block in enumerate(model.blocks):
        patch_dit_block(block, idx, storage)
    
    # 3. Initialize evidence accumulators
    n_blocks = 28
    n_steps = 250
    k_svd = 16
    
    # 跑多個 batch 累積 evidence，最後取平均
    all_l1 = np.zeros((n_blocks, n_steps))
    all_cos = np.zeros((n_blocks, n_steps))
    all_grassmann = np.zeros((n_blocks, n_steps))
    all_overlap = np.zeros((n_blocks, n_steps))
    all_sv = np.zeros((n_blocks, n_steps, k_svd))
    n_accumulated = 0
    
    # 4. Sampling loop
    for batch_idx in range(n_batches):
        # Generate noise and labels (same as sample.py)
        z = torch.randn(B, 4, 32, 32, device=device)
        y = torch.randint(0, 1000, (B,), device=device)
        
        # CFG setup
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * B, device=device)
        y = torch.cat([y, y_null], 0)
        
        # 需要手動走 sampling loop 而非直接呼叫 p_sample_loop,
        # 因為我們需要在每個 timestep 後攔截 storage
        prev_residuals = [None] * n_blocks  # 前一個 timestep 的 residual
        V_prevs = [None] * n_blocks          # 前一個 timestep 的 SVD subspace
        
        img = z.clone()
        indices = list(range(250))[::-1]  # [249, 248, ..., 0]
        
        for step_idx, i in enumerate(indices):
            t = torch.tensor([i] * z.shape[0], device=device)
            
            # 呼叫 p_sample（內部會呼叫 model.forward_with_cfg → forward → hooked DiTBlock）
            with torch.no_grad():
                out = diffusion.p_sample(
                    model.forward_with_cfg,
                    img,
                    t,
                    clip_denoised=False,
                    model_kwargs=dict(y=y, cfg_scale=4.0),
                )
            img = out["sample"]
            
            # 此時 storage 中有 28 個 block 的 residual
            for b in range(n_blocks):
                residual = storage[b]  # (2B, 256, 1152)
                
                # Evidence 1: Similarity
                if prev_residuals[b] is not None:
                    sim = compute_similarity(residual, prev_residuals[b])
                    all_l1[b, step_idx] += sim["l1_diff"]
                    all_cos[b, step_idx] += sim["cos_sim"]
                
                # Evidence 2: SVD
                svd = compute_svd_drift(residual, V_prevs[b], k=k_svd)
                all_grassmann[b, step_idx] += svd["grassmann_dist"] if not np.isnan(svd["grassmann_dist"]) else 0
                all_overlap[b, step_idx] += svd["subspace_overlap"] if not np.isnan(svd["subspace_overlap"]) else 0
                all_sv[b, step_idx] += svd["sv_spectrum"]
                
                # Update prev
                prev_residuals[b] = residual.detach().clone()  # 必須 clone，下一個 timestep storage 會被覆蓋
                V_prevs[b] = svd["V_current"]
            
            # 清理 storage（可選，下一個 timestep 會覆蓋）
            storage.clear()
        
        n_accumulated += 1
        print(f"Batch {batch_idx+1}/{n_batches} done")
        
        # 清理 prev_residuals 釋放 memory
        prev_residuals = [None] * n_blocks
        V_prevs = [None] * n_blocks
        torch.cuda.empty_cache()
    
    # 5. Average and save
    all_l1 /= n_accumulated
    all_cos /= n_accumulated
    all_grassmann /= n_accumulated
    all_overlap /= n_accumulated
    all_sv /= n_accumulated
    
    np.savez("evidence/similarity/similarity_evidence.npz",
        l1_diff=all_l1,
        cos_sim=all_cos,
        metadata={...}
    )
    np.savez("evidence/svd/svd_evidence.npz",
        grassmann_dist=all_grassmann,
        subspace_overlap=all_overlap,
        sv_spectrum=all_sv,
        metadata={...}
    )
```

### ⚠️ 關鍵實作注意事項

1. **Memory 管理是最大風險**
   - 每個 block 的 residual: `(2B, 256, 1152) × 4 bytes = 2B × 1.18 MB`
   - 若 B=4（2B=8）：每 timestep 28 blocks = ~264 MB（可控）
   - `prev_residuals` 又存了一份 = ~528 MB
   - 加上 model 本身 ~2.7 GB + diffusion 中間變數，建議 B=4 起步，在 24GB GPU 上應該可行
   - **如果爆 memory**：降低 B，或對 SVD 做 token subsampling

2. **`forward_with_cfg` 的 2B 問題**
   - `forward_with_cfg` 內部先 `half = x[:B]`，再 `combined = cat([half, half])`
   - 所以進入 `self.forward()` 時，batch 前 B 個和後 B 個的 **noise input 相同**，只有 `y` 不同
   - Hook 拿到的 `storage[b]` shape 是 `(2B, 256, 1152)`，前 B = cond，後 B = uncond
   - Evidence 計算直接在 2B 上平均，不需要拆分

3. **Timestep 的索引映射**
   - Sampling loop: `indices = [249, 248, ..., 0]`（diffusion timestep 從大到小）
   - `step_idx = 0` 對應 `timestep = 249`（最 noisy）
   - `step_idx = 249` 對應 `timestep = 0`（最 clean）
   - Evidence array `all_l1[b, step_idx]` 中 step_idx 是 sampling 順序
   - 存檔時要清楚標註這個 convention，建議同時存 `timestep_map = indices`

4. **`torch.svd_lowrank` 的 convention**
   - 返回 `(U, S, V)` where `U: (m, q)`, `S: (q,)`, `V: (n, q)`
   - Top-k right singular vectors = `V.T` → shape `(q, n)` = `(k, 1152)`
   - 請先跑 unit test 確認

5. **NaN 處理**
   - 第一個 timestep（step_idx=0）沒有 prev，similarity 和 SVD drift 都是 NaN
   - 累積 average 時要跳過 NaN 或特別處理

6. **可重現性**
   - 每個 batch 用不同但確定性的 seed: `torch.manual_seed(base_seed + batch_idx)`
   - 存檔時記錄所有 seeds

### 驗證檢查清單

實作完成後，請依序確認：

- [ ] Hook 正確性：跑一次 forward，確認 `storage` 中有 28 個 tensor，shape 都是 `(2B, 256, 1152)`
- [ ] 不改變 output：加 hook 前後，用相同 seed 跑 sampling，最終輸出圖片應完全一致（bit-identical）
- [ ] Memory 穩定：跑完整 250 steps，GPU memory 不持續增長（確認沒有 reference leak）
- [ ] Evidence 合理性：`cos_sim` 應該大部分在 0.9~1.0 之間（相鄰 timestep 的 residual 通常很相似）；接近 t=0 的 clean region 應該比 t=249 的 noisy region 更 similar
- [ ] SVD 合理性：`subspace_overlap` 大部分應接近 1.0（subspace 穩定）；singular value spectrum 應該快速衰減（前幾個 dominant）

### 超參數建議值

| Parameter | Value | Notes |
|-----------|-------|-------|
| B (per-side batch) | 4 | 2B=8 進 model，24GB GPU |
| n_batches | 16 | 共 64 張圖的 trajectories |
| n_steps | 250 | DDPM full steps |
| cfg_scale | 4.0 | 與 sample.py 預設一致 |
| k_svd | 16 | truncated SVD rank |
| base_seed | 42 | 可重現 |

### 跟 Diff-AE/LDM 版本的主要差異總結

| | Diff-AE | LDM | DiT |
|---|---|---|---|
| Architecture | UNet (encoder+decoder, skip connections) | UNet (latent) | Flat Transformer (28 blocks) |
| Blocks | 25 (mixed resolution) | 25 | 28 (homogeneous) |
| CFG | ✗ | ✗ | ✓ (batched, 2B) |
| Sampler | DDIM 100 steps | DDIM 200 steps | DDPM 250 steps |
| Residual shape | Varies by resolution | Varies by resolution | Fixed (2B, 256, 1152) |
| Cache target | Block output residual | Block output residual | `r_msa + r_mlp` (adaLN-Zero gated) |
| Hook complexity | 中（需處理 skip connections） | 中 | 低（flat sequential） |

> 請補充：
> 1. Diff-AE/LDM evidence collection 中有哪些 lessons learned（踩過的坑）
> 2. Evidence 計算中有沒有跟上述 spec 不同的 design choice（例如 similarity 是否用了不同的 normalization）
> 3. 輸出 .npz 的具體 key names，讓 DiT 版本的格式盡量對齊，方便後續 Stage 1 pipeline 共用
