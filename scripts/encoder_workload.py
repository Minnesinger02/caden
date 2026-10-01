"""CPU token accounting for the exact joint/flattened independent encoder input."""
from decision_lab.encoder import MARKER


def question_workload(row, tokenizer, mode):
    if mode not in {'joint', 'independent'}:
        raise ValueError('Unknown encoder input mode')
    criteria_sets = [row['criteria']] if mode == 'joint' else [{k: v} for k, v in row['criteria'].items()]
    texts = ['\n'.join(['State: ' + row['state'], 'Question: ' + row['question'],
                       *[f'Candidate {k}: {v} {MARKER}' for k, v in criteria.items()]])
             for criteria in criteria_sets]
    lengths = [len(ids) for ids in tokenizer(texts, padding=False, truncation=False)['input_ids']]
    return {'encoder_rows': len(lengths), 'valid_tokens': sum(lengths),
            'padded_tokens': len(lengths) * max(lengths), 'max_row_tokens': max(lengths),
            'dense_attention_cells': len(lengths) * max(lengths) ** 2}
