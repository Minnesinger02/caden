"""Fresh, balanced eight-way rule-following probe with mechanically verified gold."""
import hashlib
import itertools
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]


def flags(spec):
    a = spec['amount'] >= spec['threshold']
    if spec['family'] == 'direct':
        return a, spec['elapsed'] > spec['deadline'], spec['region'] in spec['priority_regions']
    b = not a if spec['invert_priority'] else a
    c = not b if spec['invert_hold'] else b
    return a, b, c


def generate(seed=1729, per_family=160):
    rng = random.Random(seed)
    combos = list(itertools.product([False, True], repeat=3))
    rows = []
    regions = ['north', 'south', 'east', 'west', 'central']
    for family in ['direct', 'two_hop']:
        for index in range(per_family):
            target = combos[index % 8]
            threshold = rng.randint(20, 500)
            amount = threshold + (rng.choice([0, 1, 7]) if target[0] else -rng.randint(1, 9))
            spec = {'family': family, 'threshold': threshold, 'amount': amount}
            if family == 'direct':
                priority = rng.sample(regions, 2)
                deadline = rng.randint(2, 30)
                elapsed = deadline + (rng.randint(1, 5) if target[1] else -rng.randint(0, 2))
                region = rng.choice(priority if target[2] else [r for r in regions if r not in priority])
                spec.update(deadline=deadline, elapsed=elapsed, region=region, priority_regions=priority)
                state = (f'Policy: Flag A is true when amount is at least {threshold}; otherwise false. '
                         f'Flag B is true when elapsed days is strictly greater than {deadline}; otherwise false. '
                         f'Flag C is true only for regions {", ".join(priority)}.\n'
                         f'Case: amount={amount}; elapsed days={elapsed}; region={region}.')
            else:
                spec.update(invert_priority=target[0] != target[1], invert_hold=target[1] != target[2])
                state = (f'Policy: Flag A is true when amount is at least {threshold}; otherwise false. '
                         'Flag B equals Flag A unless invert_priority is true, in which case B is the opposite of A. '
                         'Flag C equals Flag B unless invert_hold is true, in which case C is the opposite of B.\n'
                         f'Case: amount={amount}; invert_priority={str(spec["invert_priority"]).lower()}; '
                         f'invert_hold={str(spec["invert_hold"]).lower()}.')
            assigned = combos.copy()
            rng.shuffle(assigned)
            criteria = {f'route_{j}': '; '.join(f'Flag {name} is {str(value).lower()}' for name, value in zip('ABC', combo))
                        for j, combo in enumerate(assigned)}
            label = f'route_{assigned.index(flags(spec))}'
            options = list(criteria.items())
            rng.shuffle(options)
            question = 'Apply the policy to the case. Which route has exactly the resulting three flag values?'
            digest = hashlib.sha256(json.dumps({'spec': spec, 'assignment': assigned}, sort_keys=True).encode()).hexdigest()
            rows.append({'id': f'policy-{family}-{index}', 'group_id': digest, 'split': 'test',
                         'state': state, 'question': question, 'criteria': dict(options), 'label': label,
                         'policy_spec': spec, 'route_flag_assignment': assigned,
                         'family': family, 'oracle_shortlist': False})
    rng.shuffle(rows)
    return rows


def main():
    output = ROOT / 'data/policy-transfer8'
    if output.exists():
        raise FileExistsError(output)
    rows = generate()
    assert len({r['group_id'] for r in rows}) == len(rows)
    for row in rows:
        expected = tuple(row['route_flag_assignment'][int(row['label'].split('_')[1])])
        if expected != flags(row['policy_spec']):
            raise ValueError('Rule verification failed')
    output.mkdir()
    path = output / 'test.jsonl'
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    manifest = {'items': len(rows), 'families': {'direct': 160, 'two_hop': 160}, 'seed': 1729,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'task': 'eight exhaustive flag triples, randomized route assignment; no oracle shortlist',
                'gold': 'mechanical threshold, strict-deadline, membership and boolean dependency evaluation',
                'scope': 'fresh synthetic rule following; no training/calibration on these cases; not a representative real-world reasoning benchmark',
                'latent_balance': 'each family has exactly 20 instances of each target flag triple; route IDs randomized',
                'hidden_fields': 'policy_spec and route_flag_assignment are for verification only; loaders pass state/question/criteria only'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
