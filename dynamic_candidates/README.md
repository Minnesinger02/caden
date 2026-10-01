# 动态候选集合下，直接读 logits 是否足够？

这是独立子项目。主目录的 encoder-only vs Jev 是另一条研究主线。

## 研究问题与已实现内容

固定 Qwen 主干、提示、输入、精度，只改变：

1. 完整词表投影后抽取指定标签 logits（A2）。
2. 只投影指定标签的输出矩阵行（A3）。
3. 候选顺序打乱、移除正确项并增加“以上皆非”。
4. 对已有完整候选概率保留 top-k 后重新归一化，检查校准变化。

这里候选项由数据提供，Qwen 不生成候选。A/B/C 只是内部编码。代码在实际提示边界验证标签必须恰好增加一个 token；不满足就拒绝，不取多 token 标签的第一个 token 冒充完整答案。

初版仅支持 Qwen2/Qwen3 文本主干、2–26 个候选、eager/fp32。它绕过 LM 包装层调用 `lm.model`，只使用最后输入位置的 hidden state，保证 restricted 路径没有先计算完整词表。无思考文本，无 autoregressive decode。缺失候选是新任务条件，不能与原任务直接合并计分。

尚未实现：A0 生成 JSON、A1 恰好一个 token 的 generation 基线、LoRA+head 与 LoRA+标签 logits 的匹配训练消融、自动显著性检验。这些必须补齐后才能提出完整的速度/方法结论。

## 运行（都在项目主目录执行）

```bash
uv run python -m dynamic_candidates.experiment --data data/smoke.jsonl --projection full --out results/dynamic-full
uv run python -m dynamic_candidates.experiment --data data/smoke.jsonl --projection restricted --out results/dynamic-restricted
uv run python -m dynamic_candidates.experiment --data data/smoke.jsonl --projection restricted --condition shuffle --seed 42 --out results/dynamic-shuffle
uv run python -m dynamic_candidates.experiment --data data/smoke.jsonl --condition missing --out results/dynamic-missing
uv run python -m dynamic_candidates.analyze --predictions results/dynamic-restricted/predictions.jsonl --paired results/dynamic-shuffle/predictions.jsonl --top-k 3 --out results/dynamic-analysis.json
```

上述样例仅用于检查流程，首次运行会下载 Qwen3-0.6B。正式实验替换为固定测试数据，指定 `--revision MODEL_COMMIT_SHA --device cuda`。77 类 BANKING77 不能直接用于这一版标签编码，先单独准备 `--candidates 16` 的 oracle 子集，并让所有比较系统使用同一子集。

阶段计时包含分词/传输、主干、输出投影，并分别同步 GPU。同步和验证本身会增加开销；不要将这些分解计时当成生产吞吐量。完整与局部投影应在同一硬件交错多次运行，而不是只比较两个单次均值。模型读取、检查和测试均不需要 Jev key。

## 概率校准

先对独立 `split=calibration` 数据运行同一模型、提示和候选配置，得到预测；然后：

```bash
uv run python -m dynamic_candidates.analyze --predictions results/dynamic-test/predictions.jsonl --calibration results/dynamic-calibration/predictions.jsonl --out results/dynamic-calibrated.json
```

温度在预设网格上以校准集 NLL 选择，不使用测试标签拟合。入口检查校准 split 与 ID/group 不重叠；模型/提示配置一致需要对照 metadata 人工确认。NLL 下限截为 1e-15 并在指标名中注明。top-k 丢弃项仍保留为零概率，不能删除正确项落在截断区域的题。

## 正式实验协议

- 同一题至少 5 次选项排列；映射回语义 ID 后测预测翻转率与总变差距离。
- 数量、语义干扰项、标签编码、输入长度分别改变，不一次混改所有因素。
- 分开报告全候选条件和保证 gold 存在的人工短名单条件；真实候选检索另计召回率与端到端成本。
- 校准看 NLL、Brier、ECE 和风险—覆盖率；top-1 不变不代表概率可靠。
- 至少 3 个训练种子，按文档聚类 bootstrap。无训练实验记录提示与排列种子。
- 加入 A1 单 token 输出基线，避免把省去长 JSON 的收益误称为主干架构收益。

## 相关论文

- [PET (2021)](https://aclanthology.org/2021.eacl-main.20/)：提示和标签词分类。
- [Calibrate Before Use (2021)](https://proceedings.mlr.press/v139/zhao21c.html)：提示与答案偏好。
- [On Calibration of Modern Neural Networks (2017)](https://proceedings.mlr.press/v70/guo17a.html)：温度校准。
- [Pointer Networks (2015)](https://arxiv.org/abs/1506.03134)：可变候选集合的相关历史工作。
- [GLiClass (2025)](https://arxiv.org/abs/2508.07662)：应比较的动态标签分类基线。
- [Same Scores, Different Decisions (2026，预印本)](https://arxiv.org/abs/2609.27678)：Jev 请求配置稳定性评测。

这一项目的产出目标是可重复的比较，不预设直接读 logits 一定足够或一定不足。
