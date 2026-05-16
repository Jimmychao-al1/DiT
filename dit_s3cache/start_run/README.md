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

## Core conversion & tests

Stage2 expanded_mask → DiT raw timestep cache：`stage2_to_dit_cache.stage2_json_to_dit_cache_scheduler`。  
請先確認單元測試通過再解讀 FID 結果：

```bash
python -m unittest dit_s3cache.tests.test_stage2_to_dit_cache -v
```

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
