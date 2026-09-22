# QAST-EHR 实验报告

## 1. 实验结论

QAST-EHR 在标准 `train -> validation selection -> one-time test` 协议下取得 **77.03% Test Acc（7032/9129）**，超过预设的 77.00% 目标。最终结果来自单个模型、单个答案头和单个 checkpoint；没有模型集成、专家互补推理、知识蒸馏或测试集训练/选参。

## 2. 模型结构

模型使用固定预提取的 VGGish、CLIP-L/14 与 ToMe 特征，所有 QAST-EHR 可训练分支均从随机参数开始训练。主要模块包括：

1. **Query-Adaptive Evidence Router**：根据问题在视觉外观、视觉变化、音频内容和音频变化四类证据间分配权重。
2. **Motion-Sound-Question Patch Grounder**：联合运动、声音和问题相关性进行稀疏 patch 选择。
3. **Hierarchical Event Memory**：以短、中、长三个时间尺度聚合事件信息。
4. **Adaptive Position Controller**：依据音频活动、视觉运动、问题语义及时间关系动态生成位置残差，并限制实际注入比例。
5. **CMG、Temporal Reader 与 SCM**：进行跨模态交互、问题条件时序读取及语义上下文建模。
6. **单一答案头**：最终只使用一个分类头输出 42 类答案。

训练辅助目标包括 TACL、SACL、事件一致性损失、证据约束、查询多样性约束和位置使用率约束。

## 3. 实验协议

- 数据集：MUSIC-AVQA 官方训练集、验证集和测试集划分。
- 随机种子：713。
- 每个候选完整训练：15 epochs。
- 优化器：AdamW；使用 BF16 自动混合精度和梯度裁剪。
- EMA：仅对同一训练轨迹的参数进行指数滑动平均，不属于模型集成。
- 候选选择：只依据验证集指标。
- 最终测试：验证选择清单冻结 checkpoint 路径与 SHA-256 后，仅执行一次。
- 禁止项：不加载任务 checkpoint，不读取测试标签训练，不使用测试集选择候选，不融合多个模型概率。

## 4. 三个预先固定候选的验证集结果

| 候选 | Overall | Audio | Visual | Audio-Visual | AV-Temporal | 位置注入比例 | 最大查询占比 | 测试资格 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A：均衡配置 | **78.72%** | 78.54% | 87.06% | **74.81%** | **74.45%** | 5.11% | 32.42% | 通过 |
| B：强化事件/证据 | 78.61% | **79.03%** | 86.89% | 74.54% | 74.21% | 5.04% | 35.91% | 通过 |
| C：强化时序容量 | 78.24% | 77.42% | **87.30%** | 74.19% | 73.72% | 5.08% | **25.46%** | 未通过 Overall 门槛 |

资格门槛为：Overall >= 78.30%、AV-Temporal >= 72.50%、位置注入比例位于 5% 至 10%、最大查询占比不超过 70%。候选 A 的 Overall 最高，因此在查看测试结果之前被确定为最终模型。

## 5. 最终测试结果

| 指标 | 准确率 | 正确数/总数 |
|---|---:|---:|
| Audio | 76.60% | 1234/1611 |
| Visual | 85.88% | 2080/2422 |
| Audio-Visual | 72.96% | 3718/5096 |
| AV-Temporal | 69.46% | 571/822 |
| **Overall** | **77.03%** | **7032/9129** |

## 6. 最终模型与完整性信息

- Checkpoint：`D:\model\qast_ehr_a_seed713\2026-09-18-16-42-55_seed713\best.pt`
- 配置：`D:\AVQA_experiment\QA-TIGER2\QA-TIGER\reproduce_qast_recent2025_v3_77_24\experiments\qast_ehr\configs\qast_ehr_a_seed713.py`
- SHA-256：`0338c8927b2fa35f9ee9a713d2a563b114c00e371c63d66d9a9ead5ac00999e8`
- 验证集自适应位置实际注入比例：5.11%。
- 验证集最大时序查询占比：32.42%，未发生单查询塌缩。

## 7. 复现命令

在以下目录执行：

`D:\AVQA_experiment\QA-TIGER2\QA-TIGER\reproduce_qast_recent2025_v3_77_24`

### 7.1 协议审计

```powershell
C:\Users\xiaogan\anaconda3\envs\qastr2\python.exe experiments\qast_ehr\audit_protocol.py --config experiments\qast_ehr\configs\qast_ehr_a_seed713.py
```

### 7.2 重新训练三个候选并仅在验证集上选模

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File experiments\qast_ehr\run_candidates.ps1
```

### 7.3 最终测试

候选训练完成且 `validation_selection_manifest.json` 中的 `selected.eligible_for_test` 为 `true` 后执行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File experiments\qast_ehr\run_final_test.ps1
```

最终测试脚本带有一次性消费标记；已经生成 `final_test_consumed.flag` 后不会再次测试。

## 8. 结果文件

- `validation_a.json`、`validation_b.json`、`validation_c.json`：三个候选的验证结果。
- `validation_selection_manifest.json`：冻结的验证选择清单及 checkpoint 哈希。
- `final_test_results.json`：最终测试完整结果。
- `final_test_results.csv`：最终测试表格结果。
- `final_test_consumed.flag`：一次性最终测试已执行的记录。
