"""Apply publication names and fixed release links without changing evidence."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESOURCE_EN = r'''\paragraph{Caden release.} We name our compact candidate encoder Caden. Caden BANK denotes the BANKING-only recipe; Caden mixed denotes the released Caden-Encoder-Mixed checkpoints. Joint/independent and 5k labels identify ablations, not separate published mixed checkpoints. The tuned Qwen3-0.6B LoRA is a distinct baseline, not Caden's encoder architecture.
Original source is released under MIT at \url{https://github.com/Minnesinger02/caden} (commit \texttt{6ed99f473896a27051d3d44ace58411d249b525c}). The three Caden-Encoder-Mixed seeds 42/43/44 are available under Apache-2.0 at \url{https://huggingface.co/Leonard02/caden-encoder-mixed} (revision \texttt{fcf0b281309fb1720d86515f45ec5c573b63216d}). The three Qwen baseline adapters are at \url{https://huggingface.co/Leonard02/candidate-qwen3-06b-lora-mixed} (revision \texttt{c666759e18058c665477ca967c86d3e19b5e8577}); they require the pinned Qwen base and are not merged weights. Each model repository stores individual seed directories, training records and calibration. These public artifacts do not contain the private dataset/result handoff or this manuscript. The encoder requires the documented custom loader rather than a root-level Transformers pipeline. These identifying links belong to the collaborator draft; an ARR review version requires a separate anonymity review.
'''
RESOURCE_ZH = '''**Caden 发布与命名。** 本文小型候选编码器命名为 Caden。Caden BANK 指 BANKING-only 配方；Caden mixed 对应公开的 Caden-Encoder-Mixed。joint／independent 与 5k 是消融标签，不是另外发布的混合训练模型。微调 Qwen3-0.6B LoRA 是独立对照基线，不属于 Caden 编码器架构。

- 代码（MIT）：https://github.com/Minnesinger02/caden ，固定 commit `6ed99f473896a27051d3d44ace58411d249b525c`。
- Caden-Encoder-Mixed（Apache-2.0）：https://huggingface.co/Leonard02/caden-encoder-mixed ，固定 revision `fcf0b281309fb1720d86515f45ec5c573b63216d`。
- Qwen LoRA 对照（Apache-2.0）：https://huggingface.co/Leonard02/candidate-qwen3-06b-lora-mixed ，固定 revision `c666759e18058c665477ca967c86d3e19b5e8577`。

两个模型库各含种子42／43／44的独立目录、训练记录和校准文件；Qwen仅为adapter，需固定基模。Caden须使用文档中的专用loader，不能把仓库根当作Transformers pipeline。公开仓库不含私有数据／结果交接包或本论文。上述实名链接用于合作者稿；ARR审稿版须另行审查匿名性。
'''

def apply_branding(source):
    source = source.replace(r'\title{Compact Candidate Encoders', r'\title{Caden: Compact Candidate Encoders')
    source = source.replace('We study the accuracy, candidate robustness, and decision cost of a compact encoder',
                            'We study Caden, a compact candidate encoder, and its accuracy, candidate robustness, and decision cost')
    for old, new in [('Mixed encoder / system', 'Caden mixed / system'),
                     ('Encoder BANK', 'Caden BANK'), ('Encoder mixed', 'Caden mixed'),
                     ('Joint encoder, 5k', 'Caden joint, 5k'), ('Independent encoder, 5k', 'Caden independent, 5k')]:
        source = source.replace(old, new)
    if r'\paragraph{Caden release.}' not in source:
        source = source.replace(r'\begin{thebibliography}{99}', RESOURCE_EN + r'\begin{thebibliography}{99}')
    return source
