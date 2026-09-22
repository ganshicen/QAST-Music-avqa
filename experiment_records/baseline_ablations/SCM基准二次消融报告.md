# SCM-off 基准二次消融实验报告

实验协议：Seed 713，随机初始化，15 epochs；各结构按验证集选择最佳 checkpoint，结构冻结后在官方测试集只读评估一次。

## 测试集：Audio 与 Visual 子任务

| 模型 | A-Counting | A-Comparative | Audio | V-Counting | V-Location | Visual |
|---|---:|---:|---:|---:|---:|---:|
| w/o SCM（基准） | 84.96 | 61.11 | **76.16** | 83.63 | 88.16 | **85.92** |
| w/o SCM + w/o CMG | 84.56 | 62.79 | **76.54** | 82.96 | 88.08 | **85.55** |
| w/o SCM + w/o Adaptive Position | 84.96 | 61.62 | **76.35** | 83.63 | 87.84 | **85.76** |
| w/o SCM + w/o Patch Grounder | 85.94 | 59.60 | **76.23** | 83.29 | 85.96 | **84.64** |

## 测试集：Audio-Visual 子任务与总体结果

| 模型 | AV-Existential | AV-Counting | AV-Location | AV-Comparative | AV-Temporal | Audio-Visual | Overall | 相对基准 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| w/o SCM（基准） | 83.70 | 78.42 | 71.09 | 63.94 | 70.44 | **73.70** | **77.38** | +0.00 |
| w/o SCM + w/o CMG | 83.10 | 76.05 | 70.87 | 61.58 | 69.34 | **72.27** | **76.55** | -0.83 |
| w/o SCM + w/o Adaptive Position | 83.10 | 77.87 | 70.65 | 62.03 | 69.95 | **72.88** | **76.91** | -0.47 |
| w/o SCM + w/o Patch Grounder | 82.79 | 76.76 | 70.54 | 62.31 | 69.71 | **72.55** | **76.40** | -0.97 |

## 验证集结果

| 模型 | Val Overall | Val AV-Temporal |
|---|---:|---:|
| w/o SCM（基准） | 78.35 | 71.53 |
| w/o SCM + w/o CMG | 78.44 | 74.21 |
| w/o SCM + w/o Adaptive Position | 78.15 | 74.94 |
| w/o SCM + w/o Patch Grounder | 77.65 | 72.51 |

## 结论

- 在该 SCM-off 基准下，移除 CMG 后 Test Overall 下降 0.83 个百分点。
- 移除 Adaptive Position 后 Test Overall 下降 0.47 个百分点。
- 移除 Patch Grounder 后 Test Overall 下降 0.97 个百分点。
- 这是单随机种子结果；稳定性结论仍需多随机种子均值 ± 标准差支持。
