# DiT / Q-DiT S3-Cache 實驗矩陣摘要

**建立日期：** 2026-06-18  
**最後更新：** 2026-06-27  
**資料來源：** 遠端 `summary.json` / `d1_baseline_fid_qdit_*.json` 逐檔驗證  
**機器設定：** DiT-XL/2, ImageNet 256×256, CFG=1.5, seed=0, ADM FID evaluator  
**Q-DiT checkpoint：** `/home/jimmy/Q-DiT/results/003-qdit_w8a8/qdit_w8a8_state_dict.pt`

---

## 量化方法

本矩陣中 **Q-DiT W8A8** 實驗採用下列論文的 post-training quantization（PTQ）方法；**FP DiT** 為全精度 baseline，未做量化。

| 模型標籤 | 量化方法 | 論文 | 頂會 | 本實驗設定 |
|----------|----------|------|------|------------|
| Q-DiT W8A8 | **Q-DiT** | Chen et al., *Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers* | **CVPR 2025** | W8A8；DiT-XL/2；checkpoint：`003-qdit_w8a8/qdit_w8a8_state_dict.pt` |

**參考連結**

- Q-DiT 論文：[arXiv:2406.17343](https://arxiv.org/abs/2406.17343) · [CVPR 2025 Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Q-DiT_Accurate_Post-Training_Quantization_for_Diffusion_Transformers_CVPR_2025_paper.html) · [Project Page](https://q-dit.github.io/) · [GitHub](https://github.com/Juanerx/Q-DiT)

---

## 主結果（`is_primary: true`）

| ID | Model | Cache | Sampler | Steps | FID@5K | FID@50K | ρ(M) | ΔFID@5K | ΔFID@50K |
|----|-------|-------|---------|-------|--------|---------|------|---------|----------|
| FP-B-DDPM250 | FP DiT | — | DDPM | 250 | 9.192 | — | 1.00 | — | — |
| FP-S3-DDPM250 | FP DiT | S3 | DDPM | 250 | 10.032 | 3.221 | 0.373 | +9.1% | — |
| FP-B-DDIM150 | FP DiT | — | DDIM | 150 | 9.148 | 2.202 | 1.00 | — | — |
| FP-S3-DDIM150 | FP DiT | S3 | DDIM | 150 | 9.319 | 2.381 | 0.386 | +1.9% | +8.1% |
| D1-DDIM100 | Q-DiT W8A8 | — | DDIM | 100 | 9.117 | 2.290 | 1.00 | — | — |
| D3-DDIM100-K8sw3 | Q-DiT W8A8 | S3 | DDIM | 100 | 9.339 | 2.626 | 0.360 | +2.4% | +14.7% |

**D1-DDIM100 附帶 metrics（FID@5K）：** IS=245.1, sFID=34.38, Precision=0.808, Recall=0.730

**D1-DDIM100 FID@50K source：** `fid_qdit_stage2_ddim100_50k/.../0619_15_dit_baseline_full_compute/summary.json`（`dit_baseline_full_compute`，ρ=1.0；FID 離線 recovery）

---

## 全部結果

### A. FP DiT Baseline

| ID | Steps | FID@5K | FID@50K | Source |
|----|-------|--------|---------|--------|
| FP-B-DDPM250 | 250 | 9.192 | — | `fid_dit_stage2/.../0516_22_dit_baseline_full_compute/summary.json` |
| FP-B-DDIM50 | 50 | 9.086 | 2.398 | `fid_dit_stage2_ddim50/.../0518_21_...` + `0519_01_...` |
| FP-B-DDIM100 | 100 | 9.172 | — | `fid_dit_stage2_ddim100/.../0518_21_...` |
| FP-B-DDIM150 | 150 | 9.148 | 2.202 | `fid_dit_stage2_ddim150/.../0518_22_...` + `0519_20_...` |

### B. FP DiT + S3-Cache（K20/sw3/λ1.0/kmax3 blockwise）

| ID | Steps | FID@5K | FID@50K | ρ | ΔFID@5K | Source |
|----|-------|--------|---------|---|---------|--------|
| FP-S3-DDPM250 | 250 | 10.032 | 3.221 | 0.373 | +9.1% | `fid_dit_stage2/.../0517_05_...` + `0517_08_...` |
| FP-S3-DDIM50 | 50 | 10.683 | — | 0.442 | +17.6% | `fid_dit_stage2_ddim50/.../0519_00_...` |
| FP-S3-DDIM100 | 100 | 9.481 | — | 0.397 | +3.4% | `fid_dit_stage2_ddim100/.../0519_00_...` |
| FP-S3-DDIM150 | 150 | 9.319 | 2.381 | 0.386 | +1.9% | `0519_00_...` + `0519_14_...` |

### C. Q-DiT Baseline（W8A8，full compute）

| ID | Steps | FID@5K | FID@50K | Source |
|----|-------|--------|---------|--------|
| D1-DDIM50 | 50 | 9.172 | — | `dit_s3cache/fid/d1_baseline_fid_qdit_ddim50_5k.json` |
| D1-DDIM100 | 100 | 9.117 | 2.290 | FID@5K: `.../d1_baseline_fid_qdit_ddim100_5k.json`；FID@50K: `.../0619_15_dit_baseline_full_compute/summary.json` |
| D1-DDIM150 | 150 | 9.044 | — | `dit_s3cache/fid/d1_baseline_fid_qdit_ddim150_5k.json` |

FID@5K 來自 `fid_cache_sensitivity_qdit.py --baseline-only`。FID@50K 需 `sample_stage2_cache_scheduler_dit.py --baseline-only --qdit-ckpt`（與 FP `dit_baseline_full_compute` 同模式）。

### D. Q-DiT + S3-Cache（D3）

| ID | Config | Steps | FID@5K | FID@50K | ρ | ΔFID@5K | ΔFID@50K | Source |
|----|--------|-------|--------|---------|---|---------|----------|--------|
| D3-DDIM50-K20sw3 | K20/sw3 | 50 | 10.722 | — | 0.446 | +16.9% | — | `fid_qdit_stage2_ddim50/.../0611_18_...` |
| D3-DDIM50-K8sw5 | K8/sw5 | 50 | 10.692 | — | 0.360 | +16.6% | — | `.../0611_19_qdit_sweep_K8_sw5_...` |
| D3-DDIM100-K20sw3 | K20/sw3 | 100 | 9.407 | 2.622 | 0.398 | +3.2% | +14.5% | `0611_19_...` (5K) + `0618_16_...` (50K) |
| D3-DDIM100-K8sw3 ★ | K8/sw3 | 100 | 9.339 | 2.626 | 0.360 | +2.4% | +14.7% | `0611_21_...` + `0612_19_...` (50K) |
| D3-DDIM150-K20sw3 | K20/sw3 | 150 | 9.288 | — | 0.386 | +2.7% | — | `.../0611_22_...` |
| D3-DDIM150-K15sw5 | K15/sw5 | 150 | 9.224 | — | 0.353 | +2.0% | — | `.../0612_01_qdit_sweep_K15_sw5_...` |

---

## 驗證過程中發現的不一致

### 1. FP-B-DDPM250 的 FID@50K 標記錯誤（已修正）

- **原 prompt 預期：** FP-B-DDPM250 FID@50K = 3.221
- **實際：** FP-B-DDPM250 **無** FID@50K baseline run
- **3.221 真正來源：** FP-S3-DDPM250（`0517_08_baseline_p2_K20_sw3_lam1.0_kmax3_blockwise`，num_images=50000）
- **JSON 處置：** FP-B-DDPM250 `fid_50k=null`；FP-S3-DDPM250 `fid_50k=3.221`

### 2. FP-S3-DDPM250 的 ρ 近似值錯誤（已修正）

- **原 prompt 預期：** ρ ≈ 29%
- **實際（detail_stats）：** `full_compute_ratio = 0.372857`（**37.3%**）

### 3. summary.json 欄位命名

所有 run 的 FID 欄位均為 `fid_5k`，即使 `num_images=50000` 時亦同。本矩陣以 `num_images` 區分 FID@5K / FID@50K。

### 4. runs_index.jsonl 為四捨五入摘要

例如 D3-DDIM100-K8sw3 FID@50K：runs_index 寫 2.626，summary.json 精確值 2.625567。本矩陣以 summary.json 為準。

### 5. 0618 run 命名陷阱（已釐清，非 D1 baseline）

- **run：** `0618_16_qdit_baseline_p2_K20_sw3_blockwise_ddim100_50k`
- **實際類型：** **D3 S3-Cache**（K20/sw3），`full_compute_ratio=0.3975`，**不是** D1 full compute
- **命名原因：** `qdit_baseline_p2_*` 指 Stage1 config 名稱 `baseline_p2`，與矩陣 ID **D1**（`dit_baseline_full_compute`，ρ=1.0）不同
- **矩陣歸屬：** FID@50K=2.622 → **D3-DDIM100-K20sw3**

### 6. D1-DDIM100 FID@50K recovery（2026-06-20）

- **run：** `0619_15_dit_baseline_full_compute`（`dit_baseline_full_compute`，ρ=1.0）
- **FID@50K：** 2.290（summary.json 精確值 2.289759）
- **備註：** 50K 生成完成後 NPZ 打包 OOM；FID 由既有 PNG 離線 recovery

---

## 檔案位置

- **結構化 JSON：** `/home/jimmy/DiT/dit_s3cache/results/dit_experiment_matrix.json`
- **本摘要：** `/home/jimmy/DiT/dit_s3cache/results/dit_experiment_matrix_summary.md`
