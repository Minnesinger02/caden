# 后续实验：从下一个 token 到语义选项分布

2026-09-30。状态：实验设计及混合蒸馏接口已实现；尚未做预训练模型训练、教师标注或效果评测。它是现有子项目的后续方向，不改变主目录 encoder-only 实验或 LitJev 的等价推理优化。

## 问题定义与可证伪假设

“我是 奶龙”与“我是 真的 奶龙”说明：某个位置的 token 概率不等于最终语义类别的概率。奶龙还可能被切成多个 token；仅取第一个 token 会混淆共享前缀的答案。自然续写、要求输出 A/B 的提示及 Jev 的结构化输入也不是同一个条件分布。

相同权重、输入和推理设置下，停止在 prefill 或继续生成不会改变第一步 logits。未来 decode 并不会倒过来影响它。需要检验的是训练目标与输出格式是否匹配，不能把差异归因于模型“知道后面还要 decode”。

对于单 token 标签集合 C，设全词表概率为 p，候选质量为 m=sum(p[c], c in C)。候选归一化得到 p_C(c)=p(c)/m，只保留相对比例。它既不找回“真的奶龙”等序列的概率，也不能补回 top-k 已丢掉的正确候选。只改变公共分母不会改变保留候选之间的排序。

“其他”是调用者定义的语义类别，不等于 1-m。官方 Choice 文档建议在选项可能不完备时主动加入 other/none of the above，并没有把它规定成所有非候选 token 的概率总和。降低它的 logit 可能增加错误的确定性。

假设 H1：固定单 token 标签加少量格式示例能减小表达方式带来的波动。
假设 H2：在完全相同的语义候选支持集上蒸馏，可在部分任务上改善单次前向决策；教师一致性提高不必然代表真实准确率提高。
假设 H3：显式覆盖缺失正确项的训练，比恒定压低 other 更能兼顾闭集准确率与拒答能力。这些均待验证。

## 先做诊断，再训练

先选 200 个独立 gold 样本，保持源样本 group_id，构造候选完整/移除 gold 加 other 两种条件，并做候选顺序置换。另建立带有同义表达、修饰词和多 token 答案的人工审查子集；仅插入确实不改变语义的表达。变体必须留在同一数据划分中。

| 对照 | 目的 | 实现状态 |
| --- | --- | --- |
| 未训练模型，固定 A/B 标签，候选 renorm | 单次前向基础线 | 已有 evaluate |
| 同模型，加一个固定格式示例 | 区分提示格式问题与知识问题 | 待补提示版本 |
| 完整答案序列打分，并汇总预先审核的同义表达 | 检查首 token 与语义读出的差异 | 待实现，仅离线诊断 |
| 官方 Jev，相同语义 criteria | 外部教师/对照，不能推断其内部架构 | 需真实导出概率 |
| other 的 logit 减去固定偏置 | 检验朋友的直接抑制建议 | 待实现，仅消融 |

序列打分可以用 teacher forcing 批量计算，不一定需要逐 token 采样循环，但会多算候选后缀，不能记作相同成本的 prefix-only 推理。汇总同义表达必须使用明确终止符、互不重叠的完整字符串事件，报告每类别名数与未覆盖质量；有限别名集合不能冒充完整语义概率。长度归一化分数也不是原始序列概率。

诊断时额外计算完整词表才能得到 m；不要在稀疏读出的生产计时中悄悄加入它。分别记录 m、候选分布、是否选对、other 概率。m 上升但准确率未升，不能判定语义能力改善。

## 训练消融：保持同样的候选支持集

所有组共享基模、提示、数据、LoRA 设置、训练步数及随机种子；测试标签独立于教师。先跑 1,000 条训练样本的小试验，再按结果决定是否扩展。不要在动态 top-k 集合与另一个完整候选集合之间直接算 KL。

1. 无训练；
2. `token_sft`：全词表答案 token CE，直接约束输出格式；
3. `candidate_ce`：仅候选 CE；
4. `teacher_kl`：KL(q_teacher || p_student)，教师来自相同语义问题/候选；
5. `gold_kl`：α CE(gold, p_student) + (1-α) KL(q_teacher || p_student)。

第 5 组已加入训练入口。温度固定 1，α 默认 0.5；可以在开发集比较 0.25/0.5/0.75，不能用测试集选择。要求每行明确 label_source=gold，教师键必须与 criteria 完全一致，按语义键映射，不依赖字典顺序。输入 prompt 不含 gold 或教师概率。它是标准蒸馏目标的应用，不作为新算法声明。

候选 KL 只约束候选内比例，不能单独保证全词表中合法标签的总质量上升。token_sft 与 candidate_ce 的区别恰好能检查“格式对齐”和“候选判断”哪一个起作用。若教师本身就是同一个未改动学生的输出，初始 KL 近零，不会凭空带来新知识。

```bash
# 从仓库根目录；数据需含独立 gold 和已导出的教师概率。
uv run python -m prefill_renorm_sft train \
  --data data/sft-train.jsonl --objective gold_kl --gold-weight 0.5 \
  --model Qwen/Qwen3-0.6B --revision MODEL_COMMIT_SHA \
  --device cuda --dtype bfloat16 --out checkpoints/gold-kl-seed42
```

推理继续使用已有 evaluate：合并 LoRA 后，只做 prefill 和候选投影，无新增解码循环。该接口目前属于 Qwen 单题 SFT 路径，未接入 LitJev 多题包装器；LitJev 等价加速实验仍保持原权重。

## 判定标准与报告边界

同时报告 accuracy、NLL、Brier、ECE、候选置换翻转率；在“正确项缺失”组报告 other 召回率，在“正确项存在”组报告误拒率。教师 KL/一致率单独报告，不能充当 ground-truth accuracy。按源样本分组 bootstrap 置信区间，至少 3 个训练种子。

other 偏置只在开发/校准集拟合，保留偏置 0 对照，测试集保持固定。需要报告完整组和缺失组两者结果，不能以牺牲后者换来的闭集提升概括整体改进。训练与教师采集成本另计；正式延迟需同硬件、dtype、长度、题数和计时范围。当前随机小模型的工程加速数据不能用于预测蒸馏效果。

## 直接相关的原始资料

- [Surface Form Competition: Why the Highest Probability Answer Isn’t Always Right, EMNLP 2021](https://aclanthology.org/2021.emnlp-main.564/)：不同表述竞争概率质量，是此例最接近的问题背景；并不证明本实验一定有效。
- [Increasing Probability Mass on Answer Choices Does Not Always Improve Accuracy, EMNLP 2023](https://arxiv.org/abs/2305.14596)：增加合法选项概率不保证准确率改善，支持把概率质量和任务指标分开测量。
- [Distilling the Knowledge in a Neural Network, 2015](https://arxiv.org/abs/1503.02531)：软目标知识蒸馏的经典依据。
- [TypeSafe Choice 官方文档](https://docs.typesafe.ai/primitives/choice)：选项由 criteria 给定，列表不完备时可加入 other。

这些资料支持问题定位及实验设计，不支持“首次提出”、Jev 架构推断或未经测量的准确率/速度结论。
