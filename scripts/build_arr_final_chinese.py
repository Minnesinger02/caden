"""Reviewed Chinese counterpart of the complete compact ARR English master."""
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper/arr'
sys.path.insert(0, str(ROOT))
from scripts.build_chinese_translation import tables
from scripts.caden_paper_branding import RESOURCE_ZH

SECTIONS = {
    'Introduction': ('引言', '''概率决策接口可以从用户描述的候选中返回概率，无须生成解释。这一接口并不决定模型是联合编码候选、隔离候选，还是从因果语言模型读出概率。本文研究单张16GB GPU预算下，小型本地可训练编码器的领域准确率、候选变化稳定性及实际决策成本。

所有系统处理相同题目和候选payload，不将作者不同构造的榜单分数混合比较。三点观察构成分析：领域扩展显著提高监督CLINC成绩，BANKING-only迁移和合成规则执行揭示局限；候选隔离改善顺序稳定性，同时改变计算量且不保证可靠拒答；重复计时使实测速度区别于一次串行trace。复现材料保存冻结输入、逐题概率、固定版本和审计脚本，匿名打包及渲染核验作为独立门槛。'''),
    'Related Systems': ('相关系统', '''本地骨干为约66M参数的DistilBERT。因果基线使用Qwen3-0.6B和候选交叉熵LoRA，只做一次概率读出。OpenJev是使用同一固定Qwen基模的公开概率读出实现，在本研究中不是另行训练的权重。

其他系统有明显差别：Laya将ModernBERT-large与Transformer决策头及proper-score训练结合；Nano使用Ettin编码器、候选起始mask标记、小型MLP和教师分布训练；Von使用独立候选注意力、位置重置和输入相关校准，其打包实现不同于本文重复premise的消融；Kev与Decider采用更大混合语言模型骨干及各自决策训练／读出。保留官方loader、源码快照、原温度和权重版本。Jev为专有API对照，请求与响应均固定jev-1.13.0；参数量、精度、服务硬件和完整训练曝光未知。'''),
    'Task and Experimental Protocol': ('任务与实验协议', r'''每条数据包含状态、问题及候选描述动态映射。BANKING77与CLINC150主任务提供gold和七个确定性干扰项；两个数据集均为英文意图语句，没有确立说话人人口统计代表性。属于oracle shortlist选择，不是原77／150类分类、检索或CLINC OOS识别。冻结数据按规范化精确文本去重，本地来源group不跨池。完整BANKING校准／测试为1000／3079条；1k与扩展5k／8792训练池保持开发文件字节不变。

混合训练含8792 BANKING＋14997 CLINC，开发800条，校准3396条。新CLINC确认4197题排除全部300题已评分probe；读取其分数前冻结recipe和种子42、43、44。后续BANKING保留和2B补充是在已读池的探索性实验。

**本地读出。** 联合编码器在一个序列读取状态／问题和所有候选。共享标量头在候选末位标记计算 $z_i=w^\top h_i+b$，然后 $p_i=\exp(z_i)/\sum_j\exp(z_j)$，不是固定类别头。全量微调三轮，AdamW学习率 $2\times10^{-5}$，microbatch一个问题、梯度裁剪并增强候选顺序。独立消融为每项重复premise，批量编码后组合标量分数计算问题级交叉熵；骨干、训练数据、轮数和优化器匹配，但不共享premise计算。Qwen读取允许的首响应token logits，query／value LoRA rank8，训练一轮、学习率 $10^{-4}$、累积8步。容量、预训练和曝光不同，不能建立架构因果结论。

**校准与统计。** 每个本地checkpoint仅在独立校准池拟合一个正温度并原样应用：原模型用1000 BANKING，混合模型用3396 pooled。社区raw保留上游校准，额外BANKING温度另列附录。保留本地原分数，减少概率下溢的信息损失；温度不改变选择。失败在准确率中计错；有效响应NLL以 $10^{-15}$ 裁剪，并保存Brier和10个等宽bin的ECE。本地均值及样本SD描述三个已观察种子。准确率区间按来源问题配对重采样5000次，条件于这些种子；次级比较区间未作同时多重校正。

**硬件和计时。** RTX4090 Laptop（16376 MiB）、i9-13980HX、约31.6 GiB RAM、PyTorch2.8／CUDA12.8与FP32固定；主／社区环境Transformers4.57.6／5.17。一次一个GPU模型，两实际记录CPU线程。实际任务用五个随机模型block，seed1729，相同200题BANKING开发payload，B=1、预热3次；固定seed42，不按准确率选计时种子。含分词、传输、校验和概率回传，排除加载／下载，各题状态缓存为冷。决策率为题数除以同步时延总和，不是p50倒数；几何平均配对比值及5000次重采样描述此机／运行时。合成FP32／eager的B=1/8/16、L=128/256，预热5次、前向100次，排除应用开销。同硬件与形状不等于同参数或FLOPs。Encoder没有生成token速度。'''),
    'Accuracy and Domain Expansion': ('准确率与领域扩展', '''完整BANKING上，只用银行训练的encoder减微调Qwen为−0.141个百分点，配对95%区间[−0.823，0.530]，没有确立encoder更优或预定义等效性。新CLINC监督扩展增益：encoder＋17.830个百分点[16.782，18.855]，Qwen＋4.718[4.154，5.321]；混合encoder减混合Qwen为−0.929[−1.350，−0.516]。这是冻结本地确认上的监督收益，不是零样本或架构效应。

混合encoder原域保留94.42%，变化−0.173个百分点[−0.671，0.314]；混合Qwen95.00%，＋0.260[−0.195，0.704]。两区间含零，不证明稳定遗忘或等效。同一混合校准池下，测试NLL由0.1904降至0.1033（encoder）和0.0648降至0.0577（Qwen），校准后ECE为0.0080／0.0048。跨训练阶段比较同时改变校准池组成。

表中BANKING／新CLINC分别3079／4197题、相同oracle八候选；本地报告三种子均值±样本SD（百分点），社区／API各一个固定系统。混合BANKING保留、较晚2B BANKING补充及Jev API实测属于已读题目上的探索性结果；全部最终运行零失败。社区CLINC raw含原校准，Jev NLL来自舍入后的wire概率。

社区排序依任务而变：Kev和Von在BANKING表现较好，却在该CLINC构造下明显低于本地混合模型；Decider2B的CLINC准确率最高。作者卡片在Von／Decider训练中明确提及BANKING，在Nano数据metadata中列CLINC，这不是逐项重叠证明。本地精确审计仅发现一份公开Kev套件与1000校准题重合118、3079 BANKING测试题重合0，不能认证完整训练／筛选历史。'''),
    'Candidate Context and Robustness': ('候选上下文与鲁棒性', '''在相同5k训练下，independent开发准确率93.67%，joint91.67%，探索性配对收益2.00个百分点[0.17，4.00]。三种子各800次乱序比较均零翻转。独立候选分数只依赖共享premise与该候选；支持变化仍可改变归一化概率。Von在官方独立候选实现下也零观察翻转。这是200个来源问题上的稳定性，不证明所有输入鲁棒。

独立处理将每题一行变为八行：joint有效token129.6，independent345.4、padding后398.1。重复配对计时joint／independent决策率比为CONTEXT_RATIO [CONTEXT_CI]，两消融checkpoint容量与训练样本匹配。独立性不解决缺失gold识别：independent5k选显式none准确率34.50%，joint5k43.17%，BANK-only全训练池joint21.50%。该任务不是OOS或部署拒答保证。mixed模型不在冻结鲁棒性矩阵内。

表中均为开发诊断，200来源问题的原始加四种顺序，或三种支持变化；乱序flip对照各题原始预测，变体不是独立新增问题。Drop／add改变一项非gold候选；Absent删除gold并提供none，是独立任务。本地取三种子均值，社区测一个系统，全部零失败。'''),
    'Measured Cost and Engineering Ablations': ('实测成本与工程消融', r'''混合encoder／Qwen决策率比为MIX_Q_SPEED [MIX_Q_CI]，混合encoder／Kev为MIX_KEV_SPEED [MIX_KEV_CI]。区间描述五个配对block，不描述未来平台。Kev／Decider因缺少融合因果卷积／gated-delta包而使用Windows参考循环内核，比较的是此运行时，不是最佳部署上限。不同分词／prompt长度及校验路径属于系统成本；allocator峰值与含桌面占用的整卡采样不同。

表中五轮随机串行模型用同一GPU、200问题、FP32、两CPU线程与B=1。p50是各block中位数的中位数；决策／秒是block决策率中位数；比值是配对几何均值。Peak为记录到的最大allocator峰值GiB，排除桌面／驱动；prompt、模型规模和kernel不同。

同一BANK encoder权重的FP32 SDPA此前保持全部200开发选择，最大概率差 $8.73\times10^{-6}$。重复eager／SDPA决策率比为SDPA_RATIO [SDPA_CI]，小于1表示SDPA更快。BF16在两种注意力实现各改变一题，不称无损。Pointer在初始三种子没有提高准确率，NLL更差。同Qwen权重的候选词表投影在一／两问题合成读出中仅有1.033／1.115倍工程收益，与Jev API结果分开。'''),
    'Discussion and Conclusion': ('讨论与结论', '''本地训练的小型联合编码器是一种领域相关概率评分器：增加监督显著提高新第二领域确认成绩，当前运行时具有可测决策成本优势。它不能普遍替代更大决策系统。候选隔离以重复premise成本换取顺序稳定性；显式none和规则执行是分开的能力。受控上下文实验与完整系统测量给出可复现部署取舍，不建立encoder注意力方向导致更高准确率的新结论。'''),
    'Limitations': ('局限性', '''强制gold候选和benchmark描述限制外部有效性。三个种子、每社区一个checkpoint、五计时block不代表模型或硬件总体；小开发集参与recipe选择。社区训练暴露部分有声明，没有完整审计。混合／原域校准池不同，官方舍入概率限制本地与社区校准可比性。鲁棒性矩阵测BANK-only而非混合模型。本文独立候选序列不实现Von打包mask。合成形状排除应用开销，不是等FLOPs。Policy320是机械标注合成数据，不是一般推理。保留数小时训练计时跳跃，不用于连续训练速度主张。较晚Jev API对照的服务硬件未知，时延含传输，不能作为同卡计算量。提交前须完成匿名材料和ARR渲染页数验证；本机内置编译器目前平台初始化失败。'''),
    'Ethical Considerations': ('伦理考虑', '''使用公开benchmark语句和合成诊断，没有新招募人类参与者或已部署影响用户的决策。未进行专门的个人信息、冒犯性内容或人口统计公平性审计。归一化概率不证明未测任务上的事实可靠性。保存数据／模型卡、许可、源码版本与重叠证据。数据卡声明BANKING为CC BY4.0，CLINC为CC BY3.0；这些声明不证明所有底层材料权属。未来分发包须保留适用notice，并区分代码／权重条款。'''),
    'Acknowledgements': ('致谢与AI辅助披露', '''研究规划、实现与调试、实验编排、分析脚本、论文起草与修改以及中文对应稿均广泛使用OpenAI编程助手。数值主张关联真实执行的实验trace和保存的审计，不是生成的估算值。助手不列为作者。提交前，人类作者须审阅并核验完整论文、代码、引文和披露，并承担责任；本工作稿不认证该最终作者审阅已完成。'''),
    'Reproducibility Statement': ('可复现声明', '''本研究材料包含冻结数据manifest、训练recipe、checkpoint／源码版本、含失败指标、原概率／分数、GPU遥测及group／block重采样脚本。英文为母稿，中文按源哈希审阅并采用相同表格投影。本地artifact manifest记录证据哈希和审计范围。源码使用固定官方ACL匿名review模板；编译、页数及匿名发布验证是独立门槛，不能由源文件存在推断。'''),
    'Additional Diagnostics and Historical Experiments': ('附录：额外诊断与历史实验', '''初始1k scalar／pointer／Qwen开发均值81.17／80.83／86.67%；scalar5k／Qwen5k升至91.67／92.17%，scalar8792／Qwen8792为94.00／93.33%。这些开发结果参与recipe选择，不是独立测试估计。保留较高pointer头学习率及所有负结果。三个已保存种子的概率平均在开发诊断中未超过最好单成员；储存时延相加只是假设串行成本，不是实测ensemble服务。

冻结300题BANKING-only CLINC probe上encoder／Qwen均值80.67／93.33%，配对差−12.67个百分点[−16.44，−9.11]。320合成规则题含平衡直接／两层规则、八穷尽路线，均匀随机12.5%；encoder／Qwen为10.00／12.29%，Kev30.31%，Decider0.8B32.50%，较晚2B补充41.25%。该负诊断不测所有推理。在目标域不重新拟合BANK温度，迁移校准有时使社区NLL变差。Decider2B旧CLINC probe99.67%、BANKING96.62%，属于已读池补充。'''),
    'Synthetic Probability Throughput': ('附录：合成概率吞吐', '''表中每个形状五个随机配对FP32／eager block，预热5次并测100次前向，排除模型加载、分词和回传。

比较已有BANK encoder checkpoint与固定冻结Qwen基模，不是微调后任务准确率。输入GPU驻留、无padding且形状相同；两者均读八项概率，不生成解释，decoder只投影八个词表行。参数量和架构不同；处理输入token速度不是生成token速度，不定义决策／生成比值。'''),
    'Calibration, Provenance and Training Timing': ('附录：校准、来源与训练计时', '''完整BANKING本地温度后NLL为0.1865（encoder）／0.1829（Qwen），ECE0.0143／0.0093。同1000题池额外校准社区BANK结果；保留raw舍入及Kev套件118题校准重叠的限制。新CLINC的Laya／OpenJev／Von／Decider在保存官方choice处全部与概率argmax一致；Nano／Kev未保存该字段，明确未核验。固定Von权重校准标签1.2.0与新SDK响应别名1.3.0不同，以精确权重／源码版本识别。

混合encoder seed44的原wall为19268.76秒，GPU遥测缺口17615.24秒且训练log有对应elapsed跳跃。保留原计时和已完成准确率材料，不猜测原因或减去假设暂停时间，不将此总wall用于连续训练速度。成本证据来自独立五block推理计时。

32组保存的CUDA训练循环记录合计10.54小时，32组成功训练stage记录合计10.63墙钟小时；范围重合，不可相加。包含已记录的计时缺口，排除推理、下载、未记录尝试与上游预训练，两者都不是整个项目实际活跃GPU计算总量。Qwen adapter记录可训练参数1146880。另有仅读取头部的社区tensor元素清单，属于存储证据，不能直接认证模型参数数；buffers、共享权重和决策head可能不同。'''),
}

SECTIONS['Accuracy and Domain Expansion'] = (SECTIONS['Accuracy and Domain Expansion'][0], SECTIONS['Accuracy and Domain Expansion'][1] + '''

**固定API补充。** 本地recipe及题目文件冻结后，Jev-1.13.0在相同3079 BANKING、4197 CLINC、320 policy及200 BANKING开发payload上各实测一次。BANKING／CLINC概率argmax准确率JEV_BANK／JEV_CLINC%，policy为JEV_POLICY%。这是较晚探索性对照，不是新的独立模型选择确认。四HTTP请求并发下，BANKING／开发p50为JEV_LAT_BANK／JEV_LAT_DEV毫秒，含传输和服务计算，不进入本地GPU速度比表。原wire概率保存，仅在可观察小数网格限定的和误差内归一化；舍入零使裁剪NLL不同于不可获得的内部logit NLL。全部7796条最终回答有效，官方choice不在概率最大值的次数为JEV_NONMAX，平局差异另行审计。按文档费率估算7800次付费请求共JEV_COST美元，包含停机校验尝试中四条未保存回答的一次重测；596条已保存回答复用，没有自动重试或付费教师收集。''')


def main():
    evidence = json.loads((OUT / 'manuscript-evidence.json').read_text(encoding='utf-8'))
    source_path = OUT / 'main_en.tex'
    source_bytes = source_path.read_bytes()
    digest = hashlib.sha256(source_bytes).hexdigest()
    if digest != evidence['english_master_sha256']:
        raise ValueError('English changed after final evidence build; review required')
    source = source_bytes.decode('utf-8-sig')
    matches = list(re.finditer(r'\\section\*?\{([^}]+)\}', source))
    sections = {}
    for i, match in enumerate(matches):
        end = matches[i+1].start() if i+1 < len(matches) else len(source)
        for marker in [r'\begin{thebibliography}', r'\appendix', r'\end{document}']:
            location = source.find(marker, match.end(), end)
            if location >= 0:
                end = min(end, location)
        sections[match.group(1)] = source[match.end():end].strip()
    if set(sections) != set(SECTIONS):
        raise ValueError('Final English section coverage differs from reviewed Chinese rendering')
    timing_path = ROOT / evidence['source_files']['timing']['path']
    original_path = ROOT / evidence['source_files']['timing_original']['path']
    for key, path in [('timing', timing_path), ('timing_original', original_path)]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != evidence['source_files'][key]['sha256']:
            raise ValueError('Timing evidence changed since English build')
    timing = json.loads(timing_path.read_text(encoding='utf-8'))
    original = json.loads(original_path.read_text(encoding='utf-8'))
    ratios = {row['compared']: row for row in timing['paired_ratios']}
    context = evidence['matched_context_speed']
    sdpa = next(row for row in original['paired_ratios'] if row['compared'] == 'encoder-sdpa')
    substitutions = {}
    for label, record in [('MIX_Q', ratios['qwen-mixed']), ('MIX_KEV', ratios['kev'])]:
        substitutions[label + '_SPEED'] = f"{record['geometric_mean_speed_ratio']:.2f}"
        substitutions[label + '_CI'] = ', '.join(f'{value:.2f}' for value in record['ratio_ci95'])
    for label, record in [('CONTEXT', context), ('SDPA', sdpa)]:
        substitutions[label + '_RATIO'] = f"{record['geometric_mean_speed_ratio']:.2f}"
        substitutions[label + '_CI'] = ', '.join(f'{value:.2f}' for value in record['ratio_ci95'])
    jev_path = ROOT / evidence['source_files']['jev']['path']
    if hashlib.sha256(jev_path.read_bytes()).hexdigest() != evidence['source_files']['jev']['sha256']:
        raise ValueError('Jev evidence changed since English build')
    jev = json.loads(jev_path.read_text(encoding='utf-8'))
    api_time_path = ROOT / evidence['source_files']['jev_timing']['path']
    if hashlib.sha256(api_time_path.read_bytes()).hexdigest() != evidence['source_files']['jev_timing']['sha256']:
        raise ValueError('Jev timing evidence changed since English build')
    api_time = next(row for row in json.loads(api_time_path.read_text(encoding='utf-8'))['rows'] if row['system'] == 'jev-1.13.0')
    for label, task in [('BANK', 'banking'), ('CLINC', 'fresh-clinc'), ('POLICY', 'policy')]:
        substitutions['JEV_' + label] = f"{100*jev['tasks'][task]['metrics']['accuracy_all_failures_wrong']:.2f}"
    substitutions.update({
        'JEV_LAT_BANK': f"{jev['tasks']['banking']['metrics']['latency_p50_ms']:.2f}",
        'JEV_LAT_DEV': f"{jev['tasks']['banking-dev']['metrics']['latency_p50_ms']:.2f}",
        'JEV_COST': f"{jev['budget']['estimated_spent_usd']:.4f}",
        'JEV_NONMAX': str(sum(task['official_choice_not_probability_maximum'] for task in jev['tasks'].values())),
    })
    contrast_values = []
    for task, arm in [('banking', 'encoder-bank'), ('fresh-clinc', 'mixed-encoder-raw'), ('fresh-clinc', 'mixed-qwen-raw')]:
        row = next(row for row in jev['tasks'][task]['paired_comparisons'] if row['system'] == arm)
        contrast_values.append(f"{100*row['system_minus_jev_accuracy']:+.3f}个百分点[{100*row['ci95'][0]:.3f}，{100*row['ci95'][1]:.3f}]")
    title, narrative = SECTIONS['Accuracy and Domain Expansion']
    narrative = narrative.replace('不进入本地GPU速度比表', '在应用计时表中单列HTTP行，不估计同卡GPU速度比')
    SECTIONS['Accuracy and Domain Expansion'] = (title, narrative + '\n\nBANK-only encoder减Jev的BANKING差为' + contrast_values[0] +
        '；混合encoder及混合Qwen减Jev的CLINC差为' + contrast_values[1] + '及' + contrast_values[2] +
        '。来源问题配对区间条件于三个已观察本地种子及一次API实测，个别区间未作同时多重比较校正。Jev答对全部320条合成policy题，而BANK-only encoder／Qwen为10.00／12.29%，显示任务能力局限，不证明一般推理表现。')
    title, narrative = SECTIONS['Measured Cost and Engineering Ablations']
    api_text = f"相同200开发题的Jev HTTP补充p50／p95为{api_time['p50_ms']:.2f}／{api_time['p95_ms']:.2f}毫秒，均值{api_time['mean_ms']:.2f}毫秒。题数除请求时延和为{api_time['latency_rate_per_second']:.2f}/秒，测于四请求并发，不是API并发总吞吐或实测串行吞吐；此比较直接复用已保存记录，无额外付费调用。\n\n"
    narrative = narrative.replace('表中五轮随机串行模型用同一GPU、200问题、FP32、两CPU线程与B=1。', '表中本地行五轮随机串行模型用同一GPU、200问题、FP32、两CPU线程与B=1；Jev单独一轮四请求并发、无预热、服务端硬件／精度未知，含网络和服务计算。其GPU比值／显存列留空。')
    narrative = narrative.replace('决策／秒是block决策率中位数', '时延率是题数除请求时延和，本地取block决策率中位数，API不代表并发总吞吐')
    SECTIONS['Measured Cost and Engineering Ablations'] = (title, api_text + narrative)
    abstract = '''本文研究单张16GB笔记本GPU上紧凑编码器的准确率、候选鲁棒性和决策成本。在相同oracle八候选BANKING任务上，encoder三种子准确率94.60%，微调decoder94.74%，没有确立encoder准确率优势。训练扩展至23789 BANKING＋CLINC使新CLINC encoder准确率79.65%升至97.48%，decoder98.41%；同4197题的七个固定社区系统为88.52%至99.33%。五轮随机计时在此Windows运行时测得相对混合decoder的决策率比MIX_Q_SPEED、相对Kev比MIX_KEV_SPEED。独立候选编码消除观察到的顺序翻转，但重复premise计算；none识别是独立弱点。完整系统的容量、上游曝光及kernel不同。固定版本Jev API实测补充含传输成本的对照。贡献是可复现的领域准确率／成本和上下文取舍，不是架构因果排序。'''
    def fill(text):
        for key, value in substitutions.items():
            text = text.replace(key, value)
        return text
    headers = {'System': '系统', 'BANK accuracy/%': 'BANK准确率/%', 'Fresh CLINC accuracy/%': '新CLINC准确率/%',
        'CLINC raw NLL': 'CLINC原始NLL', 'Decisions/s': '决策/秒', 'Latency rate/s': '时延率/秒',
        'Mixed encoder / system [95% CI]': '混合encoder/该系统[95%区间]', 'Peak/GiB': '峰值/GiB',
        'Order flips/%': '乱序翻转/%', 'Original/%': '原候选/%', 'Drop/%': '删非gold/%',
        'Add/%': '增非gold/%', 'Absent+none/%': 'gold缺失+none/%', 'Input length': '输入长度',
        'Encoder decisions/s': 'Encoder决策/秒', 'Qwen decisions/s': 'Qwen决策/秒', 'Paired ratio [95% CI]': '配对比值[95%区间]'}
    headers['Caden mixed / system [95% CI]'] = 'Caden mixed/该系统[95%区间]'
    SECTIONS['Task and Experimental Protocol'] = (SECTIONS['Task and Experimental Protocol'][0],
        SECTIONS['Task and Experimental Protocol'][1] + '\n\n主表继续报告三个训练种子的均值±样本标准差（ddof=1）；精确种子值与样本方差见 results/seed-variance-summary.md，方差单位为百分点平方（pp²）。单发布版本的社区模型及API没有训练种子方差估计，不能以0代替缺失。计时block波动、训练种子波动和题目配对置信区间分别解释。主表速度取相同200题BANKING开发输入，不是CLINC速度；Jev为单轮HTTP四并发、含传输，时延率不是并发总吞吐。\n\n新增经典baseline为TF-IDF描述余弦和TF-IDF线性分类器（SGD log-loss、alpha1e-4、五轮、种子42/43/44）。word unigram/bigram、min_df2、最多30000特征、sublinear_tf，仅在train拟合词汇／IDF。主表baseline使用同一BANKING+CLINC混合训练文本，线性分类器另使用标签，其固定类别logit限制到请求的候选ID；不等价于Caden对任意新候选描述的评分。原始分数softmax归一化，未在test拟合温度。CPU两线程B1、五开发block；余弦缓存描述，线性计时固定seed42。CPU／GPU、容量、轮数、目标与数值精度不同，不能作等算力因果比较。这些较晚baseline为探索性补充，recipe在各自评分前冻结。')
    SECTIONS['Accuracy and Domain Expansion'] = (SECTIONS['Accuracy and Domain Expansion'][0],
        SECTIONS['Accuracy and Domain Expansion'][1] + '\n\n同混合训练的TF-IDF固定类别线性baseline在BANKING／CLINC为95.55%／97.91%，Caden mixed为94.42%／97.48%。这个有竞争力的简单baseline进一步限制encoder准确率优越性的主张；其概率校准和CPU成本不同，也不能靠描述识别未训练过的新类别。')
    SECTIONS['Additional Diagnostics and Historical Experiments'] = (SECTIONS['Additional Diagnostics and Historical Experiments'][0],
        SECTIONS['Additional Diagnostics and Historical Experiments'][1] + '\n\n**新增完整类别benchmark。** 固定DeepPavlov SNIPS variant（1400测试行，1395规范化文本group）与AG News（7600测试行），分别呈现全部七／四类别，不按gold挑干扰项。保持官方test，训练排除精确文本重叠、去重并从每类train留最多100题dev；描述仅用类别名，不用另附的LLM生成描述。Caden直接迁移已有三个混合训练checkpoint，没有新任务更新；线性baseline在目标域train监督（SNIPS12167／AG News119439题），余弦只拟合词汇／IDF。监督暴露不等，比较用于诊断迁移而非架构因果效果。新任务所有运行零失败；社区模型及付费Jev尚未在新任务评估。表中均值±SD与样本方差使用三个种子，确定性余弦不估计种子方差。速度是各任务seed42单轮p50，Caden用GPU、TF-IDF用CPU；不同于主表五block计时。')
    if r'\paragraph{Caden v2 continuation.}' in source:
        SECTIONS['Task and Experimental Protocol'] = (SECTIONS['Task and Experimental Protocol'][0],
            SECTIONS['Task and Experimental Protocol'][1] + '\n\n**Caden v2续训。** 三个种子分别从对应原混合checkpoint开始，四域各3000共12000，固定一轮、AdamW1e-5、B1、clip1、FP32/eager与随机候选顺序，含旧域replay；不使用探索性pilot权重。续训train排除所有dev/test/calibration文本group；新温度用3996独立校准题（原3396＋新两域各300）。预先固定发布门槛为三seed平均旧dev退化不超过1个百分点、两个新dev均提升。完整test是已读benchmark上的探索性重测，不用test选择seed；额外监督不同于原社区或Qwen基线。')
        SECTIONS['Additional Diagnostics and Historical Experiments'] = (SECTIONS['Additional Diagnostics and Historical Experiments'][0],
            SECTIONS['Additional Diagnostics and Historical Experiments'][1] + '\n\nv2行单独记录多领域续训，原Caden mixed行保留跨域直接迁移成绩，不将两者混为一谈。v2新增任务各训练3000条，线性baseline使用12167／119439条，目标监督仍不匹配。v2主表速度来自之后五串行block，不与原随机block建立新的配对速度比。')
        pub=json.loads((ROOT/'release-staging/caden-v2/publication-model-completed.json').read_text(encoding='utf-8'))
        SECTIONS['Reproducibility Statement'] = (SECTIONS['Reproducibility Statement'][0],
            SECTIONS['Reproducibility Statement'][1] + '\n\n当前HF仓库根为Caden v2，固定revision `'+pub['revision']+'`；旧BANKING+CLINC版本及旧温度仍可按此前commit恢复，Qwen adapter版本未改变。')
        gh_path=ROOT/'release-staging/caden-v2/publication-github-completed.json'
        if gh_path.exists():
            gh=json.loads(gh_path.read_text(encoding='utf-8'))
            SECTIONS['Reproducibility Statement'] = (SECTIONS['Reproducibility Statement'][0],SECTIONS['Reproducibility Statement'][1] + '\n\n本次更新GitHub代码固定commit `'+gh['git_commit']+'`，Issues保持关闭。')
    projected = []
    lines = ['# Caden：单GPU预算下用于概率决策的紧凑候选编码器', '',
        '> 英文母本：main_en.tex。当前紧凑稿逐节对应完整实验快照，正文表格由英文投影并翻译表头。论文PDF、页数和匿名材料仍需最终验证，尚非提交就绪稿。', '', '## 摘要', '', fill(abstract), '']
    for name, body in sections.items():
        title, text = SECTIONS[name]
        if name == 'Reproducibility Statement':
            text += '\n\n' + RESOURCE_ZH
        if name == 'Related Systems':
            text = '本文紧凑候选编码器正式命名为 Caden，混合训练发布版为 Caden-Encoder-Mixed。Qwen LoRA 是独立对照基线。\n\n' + text
        lines += ['## ' + title, '', fill(text), '']
        table = tables(body)
        if table:
            for english, chinese in headers.items():
                table = table.replace('| ' + english + ' |', '| ' + chinese + ' |')
            projected.append(table)
            lines += [table]
    bibliography = re.search(r'\\begin\{thebibliography\}\{[^}]+\}(.*?)\\end\{thebibliography\}', source, re.S).group(1)
    entries = re.findall(r'\\bibitem(?:\[[^\]]+\])?\{([^}]+)\}\s*(.*?)(?=\\bibitem|\Z)', bibliography, re.S)
    lines += ['## 参考文献', '']
    for key, text in entries:
        text = re.sub(r'\\url\{([^}]+)\}', r'<\1>', ' '.join(text.split()))
        lines.append(f'- [{key}] {text}')
    output = '\n'.join(lines) + '\n'
    if any(token in output for token in substitutions):
        raise ValueError('Unresolved Chinese numerical placeholder')
    (OUT / '中文对应稿.md').write_text(output, encoding='utf-8')
    if any(table not in (OUT / '中文对应稿.md').read_text(encoding='utf-8') for table in projected):
        raise ValueError('Table projection mismatch')
    abstract_en = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', source, re.S).group(1).strip()
    hashes = {key: hashlib.sha256(body.encode()).hexdigest() for key, body in {'abstract': abstract_en, **sections}.items()}
    (OUT / 'translation-source-lock.json').write_text(json.dumps({'section_sha256': hashes,
        'scope': 'Complete compact ARR English snapshot reviewed against section-level Chinese rendering by local research agent; no external language peer review.'}, indent=2), encoding='utf-8')
    status = {'english_master_sha256': digest, 'translated_sections': list(sections), 'table_count': len(projected),
        'references': len(entries), 'scope': 'Complete compact mother manuscript; Chinese narratives reviewed; table data projected exactly with header translations.',
        'final_submission_ready': False}
    (OUT / 'translation-status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    (OUT / 'translation-audit.json').write_text(json.dumps({**status,
        'projected_table_lines_verified': sum(sum(line.startswith('|') for line in table.splitlines()) for table in projected),
        'chinese_sha256': hashlib.sha256((OUT / '中文对应稿.md').read_bytes()).hexdigest()}, indent=2), encoding='utf-8')
    build = json.loads((OUT / 'BUILD_STATUS.json').read_text(encoding='utf-8'))
    if build['english_sha256'] != digest:
        raise ValueError('Build status belongs to another English snapshot')
    build['chinese_matches_current_english'] = True
    build['reason'] = 'Complete frozen experiments and reviewed Chinese snapshot; final anonymous packaging and rendered page verification remain, native compiler platform initialization fails.'
    (OUT / 'BUILD_STATUS.json').write_text(json.dumps(build, indent=2), encoding='utf-8')
    print(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
