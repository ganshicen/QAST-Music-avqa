# QAST 77.38 完整归档包

## 核心结果

- 模型设置：QAST-EHR w/o SCM
- Seed：713
- Test Acc：77.3798%（7064/9129）
- Audio：76.16%
- Visual：85.92%
- Audio-Visual：73.70%
- AV-Temporal：70.44%
- 训练协议：随机初始化，15 epochs，按验证集选择最佳 checkpoint，标准 train/val/test
- 三随机种子 Overall：77.27 ± 0.18
- 三随机种子 AV-Temporal：70.92 ± 0.49

## 目录说明

- `code_no_comments/src`：从 Seed 713 训练目录中的冻结 `code_snapshot.zip` 提取并移除 Python 注释与文档字符串后的精确训练代码。
- `code_no_comments/configs`：训练和推理配置的无注释副本。
- `code_no_comments/experiments/qast_ehr`：训练队列、验证、测试与消融脚本的无注释副本。
- `code_no_comments/tests`：QAST-EHR 相关测试的无注释副本。
- `checkpoints`：Seed 713、123、456 的冻结最佳验证 checkpoint。
- `training_runs`：三个种子的训练日志、最佳轮次元数据与 TensorBoard 日志。
- `experiment_records`：77.38 主实验、三种子、组件消融、TACL/SACL 消融的原始日志、JSON、CSV、Markdown 和测试消费标记。
- `tables`：统一汇总表以及所有原始 CSV/Markdown 表格。
- `data/annots/music_avqa`：MUSIC-AVQA 标注文件。
- `MANIFEST.json`：来源、协议和 checkpoint 哈希。
- `SHA256SUMS.txt`：包内文件校验值。

## Checkpoint

| Seed | 文件 | SHA256 |
|---:|---|---|
| 713 | `checkpoints/seed713_without_scm_best.pt` | `be9695ae4557a6cc5e607dc84550b2b6764be741cccfe0e57a83ab37a55054cb` |
| 123 | `checkpoints/seed123_without_scm_best.pt` | `6cefb53cb7fa832c518716e4069822c7e39bf65d6588e7c35f9d477a85c9f17c` |
| 456 | `checkpoints/seed456_without_scm_best.pt` | `373ccdf49ebbae8e2e3c637427de055d8f4d3ec09553feced9d965e1290b49c4` |

## 注意

本归档不修改原项目文件。代码副本已移除 Python 注释与文档字符串；日志和结果保持原始内容。特征文件未复制，配置仍引用原有 VGGish、CLIP-L/14 与 ToMe 特征路径。
