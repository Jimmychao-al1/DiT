# DiT Stage2 → FID@5K（`start_run`）

與 Diff-AE `QATcode/cache_method/start_run` 對齊的產物：`run_manifest.json`、`summary.json`、`detail_stats.json`、若有 cache 排程則 `scheduler_config.snapshot.json`、`run.log`，以及結果根目錄下的 `runs_index.jsonl`（append）。

## Quick start

在 **DiT 專案根目錄**：

```bash
export PYTHONPATH=$PWD    # launcher 會自動設定，手動呼叫 Python 時再設亦可
./dit_s3cache/start_run/run_fid_with_stage2_scheduler_dit.sh
```

- 預設 **FID@5000**，隨機 class、`seed=0`。
- 預設只跑 **P2 blockwise**（`baseline_p2_K20_sw3_lam1.0_blockwise`），JSON 來自 Stage2 output。
- **`--base`**：可先跑一次 `dit_baseline_full_compute`（與 cache 共用同一 `RESULTS_ROOT`）。環境變數 **`RUN_STAGE2_CACHE_BASE=1`** 會讓 launcher 自動加上 `--base`。

## 結果目錄

```
dit_s3cache/results/fid_dit_stage2/YYYYMMDD/<sch_name>/MMdd_HH_<sch_name>/
```

- 時間戳只到**小時**（`MMdd_HH`）：單次 FID@5K 預期超過一小時，同一小時內不會重跑同名 scheduler；`mkdir(..., exist_ok=False)` 可避免意外覆蓋。
- 精確啟動時間見 `run_manifest.json` 的 `run_id`（含 `%H%M%S`）。

## `time_order` 與 mask 索引

Stage1/Stage2 JSON 的 **`time_order` 欄位值為 `"ddpm_249_to_0"`**（見 `stage2_scheduler_adapter_dit.TIME_ORDER_EXPECTED`），與 DDPM 249→0 的 zone／mask 語意一致。

- **`expanded_mask[step_idx]`**：`step_idx=0` 為去噪**第一步**（對應 `sampling_timesteps[0]`），不是 DDPM 內部 step number。
- 轉成 DiT cache 時：`raw_t = sampling_timesteps[step_idx]`（model timestep，見 `stage2_to_dit_cache.py`）。

勿與舊規格草稿中的別名字串（例如 `t_T_to_t_0_step_index`）混淆；repo 內產物與驗證皆使用 `ddpm_249_to_0`。

## Core conversion & tests

Stage2 expanded_mask → DiT raw timestep cache：`stage2_to_dit_cache.stage2_json_to_dit_cache_scheduler`。  
請先確認單元測試通過再解讀 FID 結果：

```bash
python -m unittest dit_s3cache.tests.test_stage2_to_dit_cache -v
```

## 失敗追蹤

Setup（CUDA、ckpt、scheduler JSON）或 FID 任一階段失敗時，會將 `run_manifest.json` 標為 **`failed`**（含 traceback），並在 `runs_index.jsonl` append **`st: failed`**。

## CLI

```bash
python3 dit_s3cache/start_run/sample_stage2_cache_scheduler_dit.py --help
```

自訂多組 jobs：

```bash
python3 dit_s3cache/start_run/sample_stage2_cache_scheduler_dit.py \
  --job my_sch path/to/stage2_refined_scheduler_config.json
```

FID 運算沿用 `dit_s3cache/fid/`（需 CUDA、ADM evaluator 可依 `--adm-evaluator` / `--ref-batch` 調整）。
