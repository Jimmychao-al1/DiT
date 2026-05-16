"""
DiT S3-Cache Stage 1：Offline Scheduler Synthesis。

主要模組：
  stage1_scheduler_dit.py    — global shared zones + cost-based per-block k + expanded_mask
  visualize_stage1_dit.py    — Stage 1 輸出可視化
  verify_scheduler_dit.py    — scheduler_config.json 完整性驗證
  export_stage1_sweep_csv_dit.py — 將 sweep JSON 彙整為 CSV（對齊 LDM csv_exports）
"""
