# Jev 社区复现的推理优化（SFT 为可选扩展）

第三个独立子项目的主线修正为：**针对具体的 Jev 社区复现，在保持权重与任务语义的条件下，检查并优化 prefill、候选读出和归一化路径。**不预设社区多数项目缺少这些机制，也不把闭源 Jev 当作可直接修改的代码。

主目录仍是 encoder-only vs Jev；`dynamic_candidates/` 仍研究直接读 logits 的充分性。这里强调对现有复现做微小的工程优化。详见 [推理优化审计与实验计划](OPTIMIZATION_PLAN.md)。

**推理优化已落地：**已选定并固定 LitJev 版本，在不改权重、缓存规则和归一化的前提下，跳过前缀无用词表投影，并只投影每题最终位置的候选行。[实现、测试和 CPU 小模型报告](litjev_optimization/README.md)。正式模型/GPU 跑分尚未完成。下文训练说明属于此前写好的可选 SFT 辅助代码，不是本次优化的必要步骤。

**后续语义对齐方向：**针对“真的奶龙”等表达造成的首 token / 语义答案差异，新增 [实验设计与论文依据](SEMANTIC_ALIGNMENT.md)，以及 `gold_kl` 混合训练接口。效果尚待正式模型验证。

## 已有工作与研究定位

检索日期：2026-09-30。以下是作者论文或原始项目，不把项目自报成绩当成本地复测。

| 来源 | 与本项目的关系 |
| --- | --- |
| [SALSA: Single-pass Autoregressive LLM Structured Classification (2025)](https://arxiv.org/abs/2510.22691) | 最直接的先例：结构化提示、类别到 token 映射、参数高效微调，以及仅投影相关 token 的单次前向推理。不能把这一组合声称为首次提出。 |
| [LitJev](https://github.com/zhengxuyu/litjev) | 不训练，读取现成 Qwen 的候选 logits；可作为零训练基线。 |
| [AnyJev](https://github.com/nokia-applied-research/AnyJev) | 从 prefill 的 next-token 分布获得 typed decisions，并研究选项顺序和校准；属于相关工程及评测工作。 |
| [Kev](https://github.com/jaredpalmer/kev) | LoRA + 新的 pointer head；与本项目保留词表输出头有明确区别。 |
| [PEFT 官方教程](https://huggingface.co/docs/peft/quicktour) | LoRA 训练、adapter 保存和加载的方法。 |

SALSA 支持“有人做过类似机制”，不能据此说它使用过 Jev 请求或 Jev 输出做训练。Jev 风格数据本身是一种任务封装，并非新算法。可能的研究价值在于：匹配训练条件后，比较训练目标、未见候选、顺序敏感性、概率校准与实际延迟。

## 四种训练目标

1. `token_sft`：只在答案 token 上计算全词表交叉熵，相当于 prompt loss 全部遮蔽、只监督一个答案 token 的 causal LM SFT。训练仍计算最后位置的完整词表，推理只计算候选行。
2. `candidate_ce`：只在合法候选标签上计算交叉熵。它是候选分类训练，与全词表 SFT 的归一化分母不同，必须单列实验。
3. `teacher_kl`：学生候选分布拟合教师概率分布的 KL（温度固定 1）。这是蒸馏，不等同于正确标签监督，更不能把教师错误当真值。

4. `gold_kl`：独立 gold 的候选 CE 与教师 KL 加权，`--gold-weight` 默认 0.5，要求 `label_source=gold` 和完整对齐的教师概率；温度固定 1。

四者保持主干、原 LM head 和提示相同；LoRA 默认只改 q_proj/v_proj，rank=8。推理前合并 adapter，避免把未合并的 adapter 开销混进比较。每题 1 次 prefill，无 decode 循环，无新分类头。

“微改动”指改动参数少，不代表没有反向传播成本：LoRA 训练仍需穿过主干。这里也不是修改闭源 Jev 的内部实现，而是检验开放基模的 Jev 式替代实现。

## 数据从哪里来

仅有 `state + questions + criteria` 没有监督目标。可用：

- Jev 格式输入 + 独立人工/数据集 gold：SFT 或 candidate CE。
- Jev 格式输入 + 已导出的 Jev 输出：argmax 伪标签或完整概率蒸馏。
- 只有输入：转换器拒绝，不编造答案。

转换器不请求 Jev，不产生 API 费用。导出格式，每行一个完整请求（下面概率仅为人工示意）：

```json
{"id":"document-1","group_id":"document-1","split":"train","request":{"state":"I was charged twice.","questions":{"route":{"type":"choice","instructions":"Which team should handle this?","criteria":{"billing":"Payments","technical":"Software bugs"}}}},"gold":{"route":"billing"},"response":{"model":"synthetic-example-not-real-jev","answers":{"route":{"probabilities":{"billing":0.9,"technical":0.1}}}}}
```

`gold` 和 `response` 至少有一种。拆出的多题共享 `group_id`，必须整组划分，避免同一材料跨训练/测试。请只使用有权用于训练的数据。教师标注只用于训练；正式评测需要独立 gold，评测入口拒绝 `label_source=teacher`。

```bash
# 所有命令从主目录运行
uv sync --extra sft
uv run python -m prefill_renorm_sft.data --input YOUR_EXPORTED_REQUESTS.jsonl --out data/sft-train.jsonl
```

也可以直接使用主项目 JSONL 格式（独立 ground-truth label），蒸馏另需 `teacher_probabilities`。`split=train` 是训练必要条件。输入中的 gold 与教师概率不会被放入模型 prompt。

## 运行对照

```bash
# 正式实验需指定同一个 --revision MODEL_COMMIT_SHA
uv run python -m prefill_renorm_sft evaluate --data data/smoke.jsonl --out results/prefill-base

uv run python -m prefill_renorm_sft train --data data/sft-train.jsonl --objective token_sft --device cuda --dtype bfloat16 --out checkpoints/prefill-token-sft

uv run python -m prefill_renorm_sft evaluate --data YOUR_GOLD_TEST.jsonl --adapter checkpoints/prefill-token-sft --device cuda --dtype bfloat16 --out results/prefill-token-sft
```

替换 `--objective candidate_ce`、`teacher_kl` 或 `gold_kl` 做另外三组。固定数据、种子和训练预算。无训练基线也使用本子项目 evaluate，以确保提示和计时一致。默认 Qwen3-0.6B，支持 Qwen2/Qwen3 文本主干；每题 2–26 个候选，实际答案边界必须验证为单 token。初版只实现 Choice，超过长度或标签上限会拒绝，不静默截断。不是任意模型通用封装。

训练是 microbatch=1，`--accumulation 8` 梯度累积，随机重排候选但保持语义标签/教师分布对齐。默认 1 epoch、lr=1e-4，仅作起点。FP32/eager 路径用于可验证性；BF16 可选但不承诺特定显存/吞吐量，没有梯度检查点或断点恢复。

输出 adapter 与 `experiment.json`，记录版本、训练目标、种子、训练数据 SHA、耗时；评测使用主项目统一指标和逐题日志，失败不丢弃。教师标签来源有记录但数据去重/跨文件分组检查仍需在正式实验前完成。锁定训练集后再拆开发/校准/测试，不用测试标签选模型。

## 下一步实验

同一数据划分对比：未训练、token SFT、candidate CE、教师蒸馏。报告 accuracy、NLL、Brier、ECE、风险—覆盖率和本地同步延迟。分别验证未见候选、选项乱序、正确项缺失。校准可复用 `dynamic_candidates.analyze`，只在独立校准集拟合。

Jev 作为外部 API 对照时，把网络耗时与本地计算分开说明。训练不一定改善校准或 OOD 表现；必须实测，不预先宣布加速或胜出。当前不含 A1 单 token generation 基线，需要补齐后再做完整效率结论。

## 验证状态

本地测试使用随机初始化的极小 Qwen 和本地 tokenizer，不下载预训练模型：四个目标的真实 LoRA 更新、adapter 保存/合并、评测路径、gold/teacher 区分及缺失监督拒绝。正式模型训练、Jev 教师数据采集和性能比较尚未执行。
