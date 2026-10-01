# LitJev：跳过无用词表投影的实际实验

目标已选定：[zhengxuyu/litjev](https://github.com/zhengxuyu/litjev)，固定提交 `e7fb109a7466da9709028eb9c4e9f16eaeb4e2a3`。

原始快照保存在 `../vendor/litjev-e7fb109a7466da9709028eb9c4e9f16eaeb4e2a3/`，保留原 LICENSE、NOTICE 和第三方声明。源文件未修改；`upstream.json` 记录来源、压缩包哈希和所有 Python 源文件哈希，适配器加载时验证源码未变。原仓库是 Apache-2.0。源码通过 GitHub 官方 archive 下载，不使用失败的 Git 镜像。

## 找到的真实开销

原始 `src/litjev/backend.py::TransformersScorer._score`：

1. 前缀通过 CausalLM 包装层处理，虽然设了 `logits_to_keep=1`，仍输出一次完整词表；这次 logits 没有被使用，只取了 cache。
2. 后缀批次再次调用 CausalLM，没有限制 logits 位置，产生 `[问题数, 最长后缀长度, 词表大小]` 的输出。
3. 最终每题只取最后一个有效位置的候选 token 分数。

原版**已经**有共享 prefill、独立缓存分支，以及候选内归一化。此实验省的是不必要的输出投影，而不是首次引入 prefill/renorm。保留原始归一化实现和温度。

## 改动

`adapter.py::CandidateOnlyLM` 是一个可撤销包装器：

- 前缀仅调用 `lm.model`，保留原缓存，不运行词表头。
- 后缀仍用上游原来的 mask、position_ids 和缓存复制规则。
- 当原版 `_score` 索引 `output.logits[i, last_position, candidate_ids]` 时，才执行 `F.linear(h_last, W[candidate_ids], bias[candidate_ids])`。
- 所有权重保持原样，候选集合与候选归一化也不变，不做 top-k 截断。

上游 `_score` 的代码完全不变。可用以下方式接入已有模型：

```python
from prefill_renorm_sft.litjev_optimization.adapter import optimized_scorer

scorer = optimized_scorer(model, tokenizer)
raw_scores = scorer.score(state, schema)
```

`schema` 使用快照内的 `litjev.schema.DecisionSchema`。原来的 `SchemaDecisionEngine` 也可使用这个 scorer 做 System One；实验不支持 System Two/thinking。

范围：Qwen2/Qwen3 文本模型、单设备、普通 Linear 词表头。明确拒绝 Qwen3.5 混合/视觉架构、量化自定义头和多设备分片；这些需要另做适配与验证，不能泛称所有 Jev 复现都可直接套用。

## 已完成的本地实验

环境、模型配置及全部重复计时见 [原始报告](reports/cpu-random-qwen3-with-renorm.json)。

- CPU，2 个计算线程，FP32，eager attention。
- 随机初始化的极小 Qwen3：2 层、hidden size 64、词表 32768；使用人工字符 tokenizer。
- 每组 2 次预热、20 次计时；原版与优化版交错并随机改变先后顺序。
- 每题 4 个候选；1 或 4 个问题，含不同后缀长度与 padding。
- 每次请求从冷 state cache 开始，进程/模型已热；计时包含提示编译、两次主干计算、读出、CPU 传输和原始候选归一化，不含 HTTP 和模型加载。

| State 字符数 | 问题数 | 原版 p50 | 优化 p50 | p50 加速比 |
| --- | ---: | ---: | ---: | ---: |
| 64 | 1 | 15.310 ms | 11.390 ms | 1.34× |
| 64 | 4 | 54.967 ms | 25.091 ms | 2.19× |
| 256 | 1 | 19.837 ms | 15.019 ms | 1.32× |
| 256 | 4 | 69.107 ms | 37.595 ms | 1.84× |

四组全部选项一致；最大候选 logit 绝对误差 `7.46e-8` 以下，温度 0.7/1/2 下最大概率绝对误差 `2.21e-8` 以下。

4 问题案例中，实际记录到原版头输出形状 `[1,1,32768]` 和 `[4,385,32768]`；优化版只计算合计 16 个候选标量。这是头输出分配减少的证据，**不是整个模型峰值显存或 FLOPs 同倍下降的证据**。

重要限制：随机小模型的主干很小，词表投影占比与正式模型不同；不衡量任务正确率，不能推断 0.6B/4B/27B 或 GPU 的速度。GPU 峰值显存尚未测量。本项目环境是 Transformers 4.57.6 / Torch 2.14.0；上游 pyproject 要求更新的完整应用环境，这次只验证可运行的 Qwen2/Qwen3 scorer 子集，不声称上游整个应用或所有架构兼容。

早期 `cpu-random-qwen3.json` 仅计时 scorer，保留作实验记录；正文采用后来的 `with-renorm` 报告。

## 重现与下一轮正式模型测试

从项目主目录：

```bash
uv sync --extra sft --extra litjev-audit
uv run python -m unittest discover -s tests -v
uv run python -m prefill_renorm_sft.litjev_optimization.benchmark --out results/litjev-cpu-repeat.json --repeats 20

# GPU 可用后：同一 checkpoint、相同参数下比较原版与优化版
uv run python -m prefill_renorm_sft.litjev_optimization.benchmark --model Qwen/Qwen3-0.6B --revision MODEL_COMMIT_SHA --device cuda --dtype bfloat16 --state-chars 256 1024 2048 --questions 1 4 16 --repeats 50 --out results/litjev-qwen3-gpu.json
```

正式模型命令会下载指定权重；本轮没有下载大模型、没有调用 Jev API、没有执行训练。State 参数为字符长度，报告记录实际 prefix/suffix token 数，不把字符数冒充 token 数。正式数据集正确率、更多候选数量和上下文长度应另增评测，BF16 近似相等与 argmax 一致性由脚本逐组检查，不满足会失败而不是悄悄跳过。

本地测试还覆盖：Choice/Noul/Score、不同题长、独立问题与批量一致、hidden feature 读回、多次请求缓存无污染、零次完整头调用、仍恰好两次主干调用，以及不支持架构拒绝。

验证结果：项目内 17 项 unittest 全部通过；另运行未修改上游的 scoring、distribution、candidate_codes 三个测试文件，6 项通过。上游完整测试集未运行：其 prompting 测试间接依赖本环境没有的 Qwen3.5 视觉模型类，不能把部分通过写成全套通过。首次测试收集缺少 FastAPI/HTTPX，补齐测试依赖后上述三个文件通过；不为此升级或替换主项目的 Transformers 版本。
