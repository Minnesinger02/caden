# 现有 Jev 复现的 prefill / renorm 优化

## 目标

在相同权重、输入、输出定义下，对具体社区复现做小范围推理改动。SFT/蒸馏是后续可选项，不作为主实验的前提。已选择 LitJev，提交 `e7fb109a7466da9709028eb9c4e9f16eaeb4e2a3`；[实际实现与报告](litjev_optimization/README.md)已完成 CPU 随机小模型验证。

## 已核实的反例（2026-09-30 在线 main；正式实验需固定 commit）

- [LitJev 实现说明](https://github.com/zhengxuyu/litjev/blob/main/docs/how-it-works.md)：已有共享 state prefill、缓存问题分支、候选 logits 归一化，无答案 token 生成。它不能作为“尚未做 prefill / 候选归一化”的例子。文档展示抽取 output.logits，因此是否还计算了多余位置/全词表投影，应进一步检查对应版本代码与 profiler。
- [Kev 模型代码](https://github.com/jaredpalmer/kev/blob/main/kev/model.py)：已有去掉词表头的 backbone、pointer head、候选 softmax，以及 state 前缀缓存。不能再用“删除词表投影”解释对它的加速。
- [NanoJev README](https://github.com/TianyuCodings/NanoJev)：声明零输出 token 解码与候选 softmax；共享前缀推理列在 roadmap。因此应区别“已经不生成文本”和“有没有复用共享材料”。README 不能替代运行时代码审计。

这些例子不构成对整个社区的统计，但不足以支持“多数未做这两项”的前提。

## 把优化落到具体计算

| 待确认的原路径 | 可以尝试的改动 | 需要证明什么 |
| --- | --- | --- |
| 为了返回决策仍生成多个答案 token | prefill 后直接读合法候选分数 | 与原目标是否等价；不等价时单列准确率变化 |
| 每个问题重新计算同一 state | state 预计算 + 独立分支复用缓存 | 信息隔离、位置编码、缓存生命周期正确 |
| 只需 K 个标签，却投影完整词表 | 从最后必要位置只投影 K 行，再 softmax | 候选 logits 和原分布条件化结果在容差内相同 |
| 全词表 softmax 后抽取候选、再 renorm | 对相同候选 logits 直接 softmax | 数学等价，但实际收益取决于算子耗时 |
| 已经只有 K 个候选的 softmax | 检查不必要的复制、同步、重复运算或可融合算子 | 不能把本来就存在的归一化算作新优化 |

这里只在给定合法候选集合上归一化。若先截取 top-k 再归一化，通常改变分布与概率语义，不属于保持输出等价的优化。

## 最小实验协议

1. 固定目标 commit、checkpoint、精度、硬件和依赖；跑原项目现成推理路径。
2. profiler 区分预处理、主干 prefill、后续 decode（如果有）、输出投影、softmax、设备复制和序列化。
3. 分别开启每个补丁，不一次混改。检查所有候选概率的最大绝对误差、选择一致率；如果改变目标语义，则改做效果—速度对照。
4. 输入长度 256/1024/2048，问题数 1/4/16，候选数 2/8/16（按目标支持范围扩展），每组预热后重复计时。拆开缓存冷启动与命中，不能只报重复同一请求的最佳数值。
5. 同时报 p50/p95、吞吐量、峰值显存、失败率和效果。以实际端到端占比解释收益，不能从 softmax 单算子加速推出整个模型同倍加速。
6. 获得可重复收益后再考虑 SFT，用独立实验衡量训练影响；不把权重变化与纯工程优化混合归因。

已在固定 LitJev scorer 上验证去除多余词表投影。CPU 小模型结果只支持计算路径可优化与数值一致，完整预训练模型的效果和 GPU 收益仍需测量。
