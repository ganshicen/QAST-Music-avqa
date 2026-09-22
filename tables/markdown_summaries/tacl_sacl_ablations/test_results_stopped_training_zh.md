# TACL/SACL 停止训练后的标准测试结果

测试协议：Seed 713；使用停止时已经保存的最佳验证集 checkpoint；测试集仅执行推理，不参与训练或选模。

| 模型 | Audio | Visual | Audio-Visual | AV-Temporal | Overall |
|---|---:|---:|---:|---:|---:|
| w/o TACL | 76.85 | 85.88 | 72.70 | 69.46 | 76.93 |
| w/o SACL | 76.47 | 85.96 | 73.19 | 70.92 | 77.16 |

## 细分任务

| 模型 | Audio/Counting | Audio/Comparative | Visual/Counting | Visual/Location | Audio-Visual/Existential | Audio-Visual/Counting | Audio-Visual/Location | Audio-Visual/Comparative | Audio-Visual/Temporal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| w/o TACL | 84.86 | 63.13 | 83.96 | 87.76 | 82.59 | 77.87 | 69.46 | 63.03 | 69.46 |
| w/o SACL | 85.05 | 61.78 | 83.88 | 88.00 | 83.70 | 77.39 | 71.30 | 62.22 | 70.92 |

## Checkpoint

- w/o TACL: `D:\model\qast_ehr_ablation_without_tacl_seed713\2026-09-20-13-38-35_seed713\best.pt`
  - SHA256: `c4f218430d3b121239f33bbd9f594f24b6a26302d3e898292d695e089282ac81`
  - 训练状态: completed 15 epochs; best validation checkpoint
- w/o SACL: `D:\model\qast_ehr_ablation_without_sacl_seed713\2026-09-20-15-14-12_seed713\best.pt`
  - SHA256: `507e8f04c029ccc2fde151a6dcf1557ed02b6d35e2dbe1b8ef99aa491792d975`
  - 训练状态: stopped at the start of epoch 15 after epoch 14 completed; best validation checkpoint from epoch 9
