"""Render the reviewed Chinese section translations against a locked English snapshot."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper/arr'

ABSTRACT = '''本研究在单张16GB笔记本GPU上，比较紧凑联合编码器、微调因果解码器和七个固定社区发布模型的概率候选决策。相同任务提供包含正确项的八个候选描述。在3079题BANKING上，编码器与解码器三种子平均准确率为94.60%和94.74%，没有确立编码器准确率优势。将训练从8792条BANKING扩展至23789条BANKING＋CLINC，使编码器在新CLINC确认上的准确率从79.65%升至97.48%，微调解码器达到98.41%。相同4197题CLINC上，社区发布系统准确率为88.52%至99.33%。联合与独立候选上下文的开发消融揭示准确率与计算量的取舍；合成规则诊断显示本地模型规则迁移较弱。历史本机计时促成尚待完成的随机复测。本文比较容量和训练曝光不同的完整系统，不建立架构因果结论。Jev未实测，当前研究稿尚未达到可提交状态。'''

SECTIONS = {
'Question and scope': ('研究问题与范围', '''紧凑编码器能否从数量可变、由文字描述的候选集合中作出选择，并返回归一化概率？在单张16GB GPU上，它的准确率与延迟如何取舍？本文比较完整系统。注意力方向、模型规模、预训练、微调数据、读出头和运行实现没有被分别控制，因此不能建立某种架构具有因果优势的结论。'''),
'Related Work and Design Differences': ('相关工作与设计差异', '''共享类型化概率接口不意味着共享架构。本方法采用六层DistilBERT，在每个候选描述末位读取共享标量分数；状态和问题位于联合候选列表之前。Laya采用ModernBERT-large、单独训练的两层Transformer决策头、标记评分及act/escalate头，公开训练使用RLCD适当评分奖励。Von使用候选标记、随checkpoint固定的独立候选注意力mask和位置重置，并采用输入条件校准温度。OpenDecider-nano采用Ettin编码器，在候选前置MASK处使用MLP；问题、候选位于状态之前，训练目标来自开放教师的概率分布蒸馏。因此，这些系统在容量、表示位置、候选交互、目标函数、数据来源和校准方面均不同，本文按各自固定发布实现进行比较。

源码快照和模型revision随可复现材料保存。已有编码器决策头属于相关工作；本文研究的贡献是受限GPU预算下准确率、鲁棒性、校准和成本的实证取舍。不能由社区复现推断Jev尚未公开的内部结构。'''),
'Task and data': ('任务与数据', '''BANKING77包含77种银行业务意图。本研究固定Parquet镜像revision。初始实验使用1000条训练数据，以及彼此不重叠的200条开发、200条校准和200条确认数据。开发数据来自官方训练分割，确认数据来自官方测试分割；本地分割的规范化文本组互不重叠。

每题包含八个候选key及描述。确定性构造的候选集合包含金标准选项和七个随机干扰项。因此这是oracle shortlist任务，不能视为原77类BANKING77分数，也没有评测检索。编码器训练时随机候选顺序，不静默截断输入。

对固定Kev v7/decision-v7公开训练套件的审计发现，初始本地train/dev/calibration/confirmation分别重合92/1000、16/200、27/200、0/200条。匹配依据规范化精确文本或上游来源哈希；只覆盖这一套件，不能证明整个发布版的训练来源完全无重叠。因此Kev开发集比较存在已知的数据混杂。'''),
'Methods': ('方法', r'''**联合编码器。** DistilBERT在一个序列中接收状态、问题和全部候选描述。每个候选以专用标记结束，该标记最终表示为 $h_i$，共享标量头计算

$$z_i=w^	op h_i+b,qquad p_i=rac{exp(z_i)}{sum_jexp(z_j)}.$$

评分头在候选间共享，并非固定77类分类器。整个编码器使用金标准候选交叉熵训练：AdamW、学习率 $2	imes10^{-5}$、microbatch为一个问题、梯度裁剪1、三轮训练。一轮初始实验的开发准确率为78%，训练轮数只按开发集选择。

**指针头消融。** 将联合上下文化的CLS query与候选标记分别进行LayerNorm和128维投影，以 $q^	op k_i/sqrt{128}$ 评分。保持骨干学习率和数据不变，对照头学习率 $2	imes10^{-5}$ 与 $2	imes10^{-4}$。单种子仅改善一道题后，较高头学习率方案补齐三种子；所有运行及负结果均保留。

**因果解码器。** Qwen3-0.6B对首个回答位置允许的候选token评分，并在该集合内重新归一化。候选交叉熵训练query/value LoRA，rank为8、microbatch为1、累积8步、训练一轮。分词器、prompt、优化设置、revision与adapter metadata均随运行记录保存。该方案执行一次决策读出，不生成解释。

**Kev与Jev。** Kev使用固定官方checkpoint loader，保留pointer head和学习温度。0.8B发布版基于Qwen3.5-0.8B-Base、rank-16 adapter，上游数据与本地方案不同。本地直接推理使用FP32、未合并adapter、eager注意力，不使用融合内核或CUDA graphs。Jev choice API作为计划中的外部对照；没有可用固定版本、凭据和请求预算，因此不推断或填造其成绩。'''),
'Experimental protocol': ('实验协议', r'''设备为RTX 4090 Laptop GPU（16376 MiB显存）、i9-13980HX和约31.6 GiB内存。本地实验使用FP32、PyTorch 2.8.0和CUDA 12.8；主环境Transformers 4.57.6，Kev独立环境为5.17.0。每次仅运行一个GPU模型，训练种子为42、43、44，模型及数据revision固定，分割文件保存哈希。

准确率将失败计错。NLL在有效响应上以 $10^{-15}$ 裁剪概率，同时保存Brier与十个等宽bin的ECE。均值及样本标准差描述实际三个种子；开发集参与选择，因此开发统计是描述性结果，不是无偏确认估计。确认比较采用材料group配对bootstrap。

本地单题时间经过预热和CUDA同步，包含分词、数据传输、模型执行与概率回传，排除模型加载和下载。Kev还包含请求校验，每题使用冷状态缓存。这些是实现时延，不是等token FLOPs对比；HTTP延迟另列。allocator峰值与采样整卡显存不同，后者包含桌面占用。'''),
'Development results': ('开发集结果', '''同一200题开发集上的全部列举运行均零失败。多种子p50列是各次运行中位数的平均，单个发布模型及冻结基模不报告训练种子标准差。

标量编码器均值81.17%，较高头学习率pointer为80.83%，后者NLL更差，不能声称新评分头改善了结果。本地候选CE解码器为86.67%。Kev更高的开发准确率体现整体系统能力，但已确认开发重叠、不同训练数据及模型规模使其不能用于隔离架构效应。

Kev本地直接推理中位数112.16 ms、p95为118.22 ms，HTTP中位数128.19 ms。两接口全部argmax一致，归一化概率最大差0.000185。日志显示因果卷积和gated delta规则采用PyTorch参考路径；这些时间描述当前Windows实现，不代表Kev最佳部署延迟。'''),
'Fixed-shape throughput and token semantics': ('固定形状吞吐及token口径', '''概率输出不生成自回归token，因此报告决策/秒和处理输入token/秒。GPU驻留的非padding合成输入使用相同batch与长度、FP32、eager注意力、八个候选概率，预热5次、重复前向20次。这些测量排除分词、传输与回传，与真实任务服务时延分开；编码器和解码器的参数量仍不同。

冻结Qwen基模另有历史生成实验：使用KV cache进行贪心解码，重复三次、每次恰好64个新token。batch=1、输入128 token时含prefill和decode吞吐为44.18生成token/秒；batch=8合计329.68生成token/秒。编码器的生成token吞吐没有定义，本文不计算跨任务的“编码器决策吞吐/解码器生成吞吐”比例。后续主速度实验关注决策吞吐。'''),
'Engineering readout experiment': ('读出工程优化实验', r'''在相同Qwen3权重上，固定本地LitJev实现的读出优化仅投影目标suffix位置需要的候选词表行。64字符合成状态、一个和两个问题的随机配对GPU计时加速分别为1.033和1.115倍。最大概率差约 $1.6	imes10^{-6}$，全部argmax一致。它是合成prompt上的数值与性能实验，不是任务准确率对照或Jev API测量；更大工作负载和BF16尚待验证。'''),
'Attention Implementation and Precision Ablation': ('注意力实现与精度消融', r'''使用同一8792条训练所得标量编码器权重，对照eager与原生SDPA注意力、FP32与BF16；训练与读出不变，四组使用相同200题开发集，均零失败。

FP32 SDPA全部argmax保持一致，最大概率差 $8.73	imes10^{-6}$。BF16 eager与BF16 SDPA各改变一个预测，最大概率差分别为0.2125和0.0854，因此较快BF16并不保持概率一致。FP32 SDPA是当前无观测开发预测漂移的加速候选。表中仍是单次运行计时，需要随机多轮复测后才能给出稳健部署速度结论。'''),
'Full Held-Out Confirmation and Calibration': ('完整留出确认与校准', r'''在读取测试分数之前，冻结1000条独立校准数据和3079条去重官方测试数据，之前200条小样本池保持为不变子集。本地8792条训练、开发、校准和测试group不重叠，保留全部三个训练种子，不使用测试成绩选择模型、prompt或种子。这仍是oracle八候选任务，不能证明社区模型从未见过这些公开数据。

每个checkpoint只在校准集拟合一个正温度，并原样应用于测试。保留编码器和解码器原始分数以避免概率下溢的信息损失，温度校准不改变argmax。只报告完整三种子比较，缺失组保持待完成。

公开SDK可能舍入概率，Laya和OpenJev保留四位小数接口值。保存原始响应，仅在已知舍入精度可解释的总和误差内归一化，NLL仍受舍入限制。初始严格总和校验导致的adapter拒绝作为接入诊断保留，不当作模型成绩。精度与注意力实现分别记录；时间和模型差异不隔离架构因果关系。

编码器减Qwen的准确率差为−0.141个百分点，材料group配对bootstrap的95%区间为[−0.823，0.530]。该区间条件于实际三个种子、跨零，不能证明编码器准确率优势，也没有预先定义的等效性结论。'''),
'Released Community Models on the Full Test': ('社区发布模型完整测试', '''固定社区checkpoint通过官方loader使用相同3079题和候选集合，不在本地BANKING训练池再次微调。每个模型测一个发布checkpoint，不是多种子重新训练。保留发布模型原校准，在同1000条独立校准题上再拟合一个温度。社区接口暴露概率，部分已舍入，而本地模型可读取raw scores，限制了校准比较的可比性。

审计的Kev公开套件与完整校准集重合118/1000条、与完整测试集重合0/3079条，仅覆盖该套件，不能证明全部训练或模型选择无重叠。Laya、OpenJev、Von、Decider四位小数接口概率的argmax在完整测试上与官方choice均无差异。Von权重校准文件标识von-1.2.0，而新SDK响应标签为1.3.0，因此实验按固定HF revision识别权重。

这些比较不支持编码器普遍准确率优势。紧凑本地编码器针对银行域训练，需要迁移、规则执行和上下文消融才能界定适用范围；Kev和Decider的Windows参考循环内核也限制其最优部署速度结论。'''),
'Candidate-Context Ablation': ('候选上下文消融', '''在相同5000条训练、三个种子、优化器及三轮训练下，对照joint与独立候选序列。独立方案为八个候选分别重复状态和问题，批量编码后用同一标量头读取末位标记，再组合成问题级交叉熵。它不共享premise计算，也不等于Von的打包mask实现。CPU结构测试验证候选乱序或加入候选不改变存活候选的原始logit；改变候选支持时归一化概率可以变化。GPU乱序测试尚未执行。

independent开发均值93.67%，joint为91.67%；配对差2.00个百分点，95%区间[0.17，4.00]，条件于已观测种子，是探索性开发结果而非测试确认。每题encoder行从1增至8，有效token从129.6增至345.4，padding后398.1；各次p50均值从3.11增至4.49 ms。因此观测开发收益有计算和延迟成本。注意力单元数只是形状代理量，不是实测FLOPs。'''),
'Transfer and Policy Diagnostics': ('迁移与规则诊断', '''冻结300题CLINC150 in-scope oracle八候选迁移集，不进行本地CLINC微调。另构造320题规则诊断，使用八个穷尽flag路线，要求数值阈值比较或两层Boolean依赖；金标准flag与路线经过机械校验。均匀随机准确率12.5%，该合成诊断不能代表全部真实推理。目标任务不参与训练或校准，全部评测零失败。

只用BANKING训练的编码器CLINC准确率80.67%，Qwen为93.33%，配对差−12.67个百分点，95% group-bootstrap区间[−16.44，−9.11]，条件于三个种子。CLINC规范化精确文本与本地全部BANKING来源池不重叠，但不证明社区预训练或模型选择无重叠。编码器和Qwen规则准确率10.00%与12.29%，没有可靠可迁移规则执行的证据；Kev30.31%、Decider-0.8B32.50%，也远未达到可靠正确性。

BANKING拟合的温度直接应用、不在目标集重拟合，选择保持不变；编码器CLINC NLL由1.203降至0.660，Qwen由0.281降至0.232，但无法补回任务能力。社区NLL在校准域迁移后有时变差，完整ECE与direct/two-hop结果随trace保存。

这些观察促成另行预登记的监督BANKING＋CLINC领域扩展，训练23789条，4197条新CLINC测试题排除已评分300题。这不是零样本迁移；已评分BANKING测试上的后续结果属于探索性结果。'''),
'Supervised Domain Expansion and Fresh CLINC Confirmation': ('监督领域扩展与新CLINC确认', '''在读取新CLINC确认分数前，冻结23789条混合训练数据（8792 BANKING＋14997去重CLINC）、800条开发数据和3396条校准数据。4197题测试排除已评分的300题CLINC probe。原BANKING-only和混合模型均用种子42、43、44，目标题目及候选payload相同。这是监督领域扩展，不是零样本迁移；仍是150个in-scope意图中含gold的八候选任务，不测OOS。

编码器准确率由79.65%升至97.48%，Qwen由93.69%升至98.41%。按来源问题配对bootstrap，提升分别为17.830个百分点，95%区间[16.782，18.855]，以及4.718个百分点，[4.154，5.321]。混合encoder减混合Qwen为−0.929个百分点，[−1.350，−0.516]。区间条件于三个已观察种子，不是种子总体区间或架构因果估计；参数量、预训练及三轮对一轮的训练曝光不同。

表中本地模型取三个种子均值，社区模型各测一个固定发布版本，全部零失败。社区raw保留原checkpoint校准，未额外拟合CLINC温度。混合本地模型使用同一独立校准池：encoder NLL从0.1904降至0.1033，Qwen从0.0648降至0.0577，选择不变。BANKING-only温度使用1000条BANKING校准数据，校准来源差异使跨阶段概率质量变化不能只归因训练或架构。社区上游训练与模型选择暴露未获认证。Decider2B在该任务达到99.33%，Kev88.90%，Von90.40%；这是特定候选构造下的系统比较，不是普遍能力排名。

配图左侧为三种子均值与样本SD，中间为条件于这些种子的来源问题配对区间，右侧为同一独立池的混合模型校准。已评分3079题BANKING的探索性原域复测中，encoder为94.42%，相对原模型变化−0.173个百分点，[−0.671，+0.314]；Qwen为95.00%，变化+0.260个百分点，[−0.195，+0.704]。两区间均含零，既不确立系统性遗忘，也不证明等效。条件仍是三个观察种子；新旧校准池不同，校准后概率质量变化不能隔离训练效应。'''),
'Discussion': ('讨论', '''紧凑编码器有较低本机决策延迟及显存需求，完整BANKING准确率接近调优解码器，配对区间不证明准确率优势，而银行域训练后的CLINC迁移显著较弱。所选系统不能代表全部双向编码器或概率解码器，结果描述部署取舍而不是普遍排名。

已完成1000、5000、8792条嵌套数据规模实验，开发文件字节保持不变，显示增加金标准数据如何改变准确率。两方案各组使用相同样本，但encoder三轮与decoder一轮不等训练曝光。CLINC probe不测OOS，也不证明社区预训练或模型选择未见过该数据。预登记混合领域训练与新CLINC本地及社区确认已完成，扩展后的encoder在新监督任务达到97.48%。

完整BANKING确认、独立校准和冻结迁移诊断已完成。候选乱序/缺失GPU检查、进一步的大容量迁移诊断和随机多轮计时尚未完成。Jev还需固定可用版本与请求预算。完成剩余实验和渲染页数核验前，本文仍是研究草稿。'''),
'Conclusion': ('结论', '''紧凑joint编码器有较低历史本机决策延迟，随机复测仍待完成。完整BANKING比较未确立其相对微调解码器的准确率优势。BANKING-only的CLINC迁移明显更弱，监督领域扩展则使新CLINC准确率升至97.48%，低于微调解码器98.41%。探索性原域保留区间含零。pointer头未提高三种子均值，本地规则表现接近随机。这些发现描述完整系统的领域相关准确率与成本取舍。候选鲁棒性、进一步大容量迁移诊断和随机计时仍在进行，当前ARR稿件尚未达到可提交状态。'''),
'Limitations': ('局限性', '''任务强制正确项在八候选中，不测检索、原77类BANKING或150类CLINC基准，也不测CLINC OOS。小开发集参与模型选择，其成绩属于探索性；新CLINC确认排除已评分probe并报告全部冻结臂。架构、参数量、预训练、训练曝光及运行实现不同，新旧本地校准池也不同。社区上游暴露未获认证；Kev一份公共套件存在已确认开发重叠，与一套件零重叠不证明全部训练无重叠。合成固定形状吞吐排除应用开销，Windows参考内核限制Kev部署结论。保留已观察训练计时中断证据，不将其用于连续训练速度结论。概率输出没有生成token速度，Jev未实测。进一步大容量迁移诊断、候选鲁棒性和随机计时尚未完成。'''),
'Ethical Considerations': ('伦理考虑', '''本研究使用公开基准文本，没有收集人体参与实验数据，没有将决策部署到影响用户的场景。模型在评测范围之外可能自信地出错，因此衡量校准，而不把有界概率输出视为事实正确性。数据集和checkpoint来源、许可证及可能重叠需要在发布材料中持续记录。'''),
'Reproducibility': ('可复现性', '''本地材料记录冻结分割哈希、checkpoint和源码revision、训练telemetry、逐题概率和实际失败；独立校准及配对group重采样脚本保留。匿名补充材料打包和最终渲染页数核验尚在进行。'''),
}


def plain_cell(cell):
    cell = cell.strip().replace(r'\%', '%').replace(r'\_', '_').replace(r'\&', '&')
    headers = {'System': '系统', 'Runs': '运行数', 'Acc. (%)': '准确率（%）', 'SD (pp)': '种子标准差（百分点）',
               'p50 (ms)': 'p50（ms）', 'Batch': '批量', 'Input tokens': '输入token',
               'Encoder decisions/s': 'Encoder决策/秒', 'Qwen decisions/s': 'Qwen决策/秒', 'Ratio': '比值',
               'Implementation': '实现', 'Flips': '预测变化数', 'Model and variant': '模型及变体',
               'Seeds': '种子数', 'SD': '种子标准差', 'Model': '模型', 'Raw NLL': '原始NLL',
               'Postcal NLL': '后校准NLL', 'Postcal ECE': '后校准ECE', 'Context': '候选上下文',
               'Valid tokens': '有效token', 'Padded tokens': 'padding后token', 'Policy': '规则诊断',
               'Direct': '直接规则', 'Two-hop': '两层依赖', 'joint': '联合', 'independent': '独立',
               'encoder raw': 'encoder 原始', 'encoder calibrated': 'encoder 校准',
               'qwen raw': 'qwen 原始', 'qwen calibrated': 'qwen 校准'}
    return headers.get(cell, cell.replace('--', '—'))


def tables(body):
    result = []
    for tabular in re.finditer(r'\\begin\{tabular\}\{[^\n]+\}(.*?)\\end\{tabular\}', body, re.S):
        text = re.sub(r'\\(?:toprule|midrule|bottomrule)', '', tabular.group(1))
        rows = [[plain_cell(c) for c in line.split('&')] for line in re.split(r'\\\\', text) if '&' in line]
        if not rows or any(len(row) != len(rows[0]) for row in rows):
            raise ValueError('Unexpected English table structure')
        result += ['| ' + ' | '.join(rows[0]) + ' |', '| ' + ' | '.join(['---'] * len(rows[0])) + ' |']
        result += ['| ' + ' | '.join(row) + ' |' for row in rows[1:]]
        result += ['']
    return '\n'.join(result)


def main():
    source_path = OUT / 'main_en.tex'
    raw = source_path.read_bytes()
    source = source_path.read_text(encoding='utf-8-sig')
    abstract = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', source, re.S).group(1).strip()
    matches = list(re.finditer(r'\\section\*?\{([^}]+)\}', source))
    sections = {}
    for i, match in enumerate(matches):
        end = matches[i+1].start() if i + 1 < len(matches) else source.index(r'\begin{thebibliography}')
        sections[match.group(1)] = source[match.end():end].strip()
    if set(sections) != set(SECTIONS):
        raise ValueError('English section coverage changed; review and extend Chinese translations')
    hashes = {name: hashlib.sha256(body.encode()).hexdigest() for name, body in {'abstract': abstract, **sections}.items()}
    lock_path = OUT / 'translation-source-lock.json'
    lock = json.loads(lock_path.read_text(encoding='utf-8'))
    if lock['section_sha256'] != hashes:
        raise ValueError('English narrative or table changed since translation review; refresh Chinese and source lock explicitly')
    lines = ['# 单GPU预算下用于概率决策的紧凑候选编码器', '',
             '> 英文母本：main_en.tex。本稿逐节对应当前锁定英文版本；所有表格单元直接由英文母本提取，保留原模型标识和指标缩写。实验与最终版式仍未全部完成，本文不是提交就绪稿。', '',
             '## 摘要', '', ABSTRACT, '']
    for name, body in sections.items():
        title, translated = SECTIONS[name]
        lines += ['## ' + title, '', translated, '']
        table = tables(body)
        if table:
            lines += [table]
    bibliography = source[source.index(r'\begin{thebibliography}'):source.index(r'\end{thebibliography}')]
    entries = re.findall(r'\\bibitem(?:\[[^\]]+\])?\{([^}]+)\}\s*(.*?)(?=\\bibitem|\Z)', bibliography, re.S)
    lines += ['## 参考文献', '']
    for key, text in entries:
        text = re.sub(r'\\url\{([^}]+)\}', r'<\1>', ' '.join(text.split()))
        lines.append(f'- [{key}] {text}')
    (OUT / '中文对应稿.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    status = {'english_master_sha256': hashlib.sha256(raw).hexdigest(), 'translated_sections': list(sections),
              'table_count': len(re.findall(r'\\begin\{tabular\}', source)), 'references': len(entries),
              'scope': 'reviewed section-level Chinese rendering of this locked English snapshot; tables copied from master; future English changes require renewed translation review',
              'final_submission_ready': False}
    (OUT / 'translation-status.json').write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
