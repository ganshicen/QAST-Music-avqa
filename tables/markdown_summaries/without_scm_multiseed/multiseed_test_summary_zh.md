# QAST-EHR 三随机种子标准测试汇总

协议：Seed 713/123/456，随机初始化，完整训练 15 epochs，按验证集选择最佳 checkpoint；测试集仅用于最终一次推理评估。

| 指标 | Seed 713 | Seed 123 | Seed 456 | 均值 ± 样本标准差 |
|---|---:|---:|---:|---:|
| Audio/Counting | 84.96 | 85.15 | 85.15 | 85.09 ± 0.11 |
| Audio/Comparative | 61.11 | 61.28 | 61.28 | 61.22 ± 0.10 |
| Visual/Counting | 83.63 | 83.96 | 84.13 | 83.90 ± 0.26 |
| Visual/Location | 88.16 | 88.08 | 88.65 | 88.30 ± 0.31 |
| Audio-Visual/Existential | 83.70 | 83.10 | 83.30 | 83.37 ± 0.31 |
| Audio-Visual/Counting | 78.42 | 77.31 | 78.18 | 77.97 ± 0.58 |
| Audio-Visual/Location | 71.09 | 70.54 | 70.98 | 70.87 ± 0.29 |
| Audio-Visual/Comparative | 63.94 | 62.67 | 62.58 | 63.06 ± 0.76 |
| Audio-Visual/Temporal | 70.44 | 70.92 | 71.41 | 70.92 ± 0.49 |
| Audio | 76.16 | 76.35 | 76.35 | 76.29 ± 0.11 |
| Visual | 85.92 | 86.04 | 86.42 | 86.13 ± 0.26 |
| Audio-Visual | 73.70 | 73.02 | 73.41 | 73.38 ± 0.34 |
| Overall | 77.38 | 77.06 | 77.38 | 77.27 ± 0.18 |
| AV-Temporal | 70.44 | 70.92 | 71.41 | 70.92 ± 0.49 |

## Checkpoint 证据

- Seed 713: `D:\model\qast_ehr_ablation_without_scm_seed713\2026-09-19-13-43-55_seed713\best.pt`
  - SHA256: `be9695ae4557a6cc5e607dc84550b2b6764be741cccfe0e57a83ab37a55054cb`
- Seed 123: `D:\model\qast_ehr_ablation_without_scm_seed123\2026-09-20-10-08-45_seed123\best.pt`
  - SHA256: `6cefb53cb7fa832c518716e4069822c7e39bf65d6588e7c35f9d477a85c9f17c`
- Seed 456: `D:\model\qast_ehr_ablation_without_scm_seed456\2026-09-20-11-34-24_seed456\best.pt`
  - SHA256: `373ccdf49ebbae8e2e3c637427de055d8f4d3ec09553feced9d965e1290b49c4`
