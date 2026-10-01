"""Check current evidence and manuscript snapshots; never certify unmeasured gates."""
import hashlib
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def main():
    evidence = read('paper/arr/manuscript-evidence.json')
    digest = sha('paper/arr/main_en.tex')
    require(digest == evidence['english_master_sha256'], 'English evidence mismatch')
    for record in evidence['source_files'].values():
        require(sha(record['path']) == record['sha256'], 'Manuscript input changed')
    translation = read('paper/arr/translation-audit.json')
    build = read('paper/arr/BUILD_STATUS.json')
    require(translation['english_master_sha256'] == digest == build['english_sha256'], 'Translation mismatch')
    require(translation['chinese_sha256'] == sha('paper/arr/中文对应稿.md'), 'Chinese content changed')
    require(len(translation['translated_sections']) == 14 and translation['table_count'] >= 4
            and translation['references'] >= 12 and translation['projected_table_lines_verified'] >= 52,
            'Incomplete bilingual projection')
    require(build['chinese_matches_current_english'] and not build['submission_ready']
            and not build['pdf_compile_verified'], 'Unverified submission gate misrepresented')
    source = (ROOT / 'paper/arr/main_en.tex').read_text(encoding='utf-8')
    require(r'\author{Anonymous ACL submission}' in source and r'\usepackage[review]{acl}' in source,
            'Manuscript review settings changed')
    require(r'\section*{Acknowledgements}' in source and 'not certified by this working draft' in source,
            'AI/human-review disclosure missing')
    diagnostic = read('paper/arr/native-compile-diagnostic.json')
    require(diagnostic['english_sha256'] == digest and diagnostic['kind'] == 'compile-failed',
            'Compiler diagnosis belongs to another source')
    queue = read('handoff/remaining-registered-experiments/completed.json')
    require(len(queue['completed']) == len(queue['stages']) == 6
            and all(item['exit_code'] == 0 for item in queue['completed']), 'Registered queue incomplete')
    fresh = read('results/fresh-clinc-summary.json')
    require(fresh['items'] == 4197 and not fresh['pending'] and len(fresh['rows']) == 15
            and all(row['failures_total'] == 0 for row in fresh['rows']), 'Fresh confirmation incomplete')
    robust = read('results/candidate-perturbation-summary.json')
    require(len(robust['traces']) == 38 and len(robust['rows']) == 11
            and robust['source_questions'] == 200, 'Robustness evidence incomplete')
    timing = read('results/replicated-task-fp32-8792/summary-mixed-reference.json')
    require(timing['blocks'] == list(range(5)) and timing['items'] == 200 and len(timing['rows']) == 14
            and all(len(row['decision_rates_by_block']) == 5 for row in timing['rows']), 'Task timing incomplete')
    synthetic = read('results/replicated-fp32-eager-8792/summary.json')
    require(len(synthetic['rows']) == 6 and all(row['blocks'] == 5 for row in synthetic['rows']),
            'Fixed-shape timing incomplete')
    tests = (ROOT / 'handoff/tests-final-registered-experiments.log').read_text(encoding='utf-8')
    require('Ran 63 tests' in tests and tests.rstrip().endswith('OK'), 'Final test record missing')
    api_tests = (ROOT / 'handoff/tests-jev-budget-safety.log').read_text(encoding='utf-8-sig')
    require('Ran 5 tests' in api_tests and api_tests.rstrip().endswith('OK'), 'API financial safety test record missing')
    jev = read('results/jev113-comparison-summary.json')
    require(jev['model'] == 'jev-1.13.0' and sum(t['items'] for t in jev['tasks'].values()) == 7796
            and all(t['metrics']['failures'] == 0 for t in jev['tasks'].values())
            and jev['budget']['attempted_paid_requests_including_probe'] == 7800
            and jev['budget']['unknown_usage_requests'] == 0
            and jev['budget']['estimated_spent_usd'] < .5, 'Pinned API comparison incomplete or over budget')
    v2_path=ROOT/'results/caden-multidomain-v2/integrity-audit.json'
    if v2_path.exists():
        v2=read('results/caden-multidomain-v2/integrity-audit.json')
        require(v2['runs_verified']==12 and v2['questions_verified']==48828 and v2['train_heldout_groups_disjoint'] and v2['timing_blocks_verified']==5, 'V2 empirical audit incomplete')
        require(sha('results/caden-multidomain-v2/completed.json')==v2['completed_sha256'], 'V2 result changed after audit')
        v2_tests=(ROOT/'handoff/tests-caden-v2-working.log').read_text(encoding='utf-8-sig')
        require('Ran 72 tests' in v2_tests and v2_tests.rstrip().endswith('OK'), 'V2 full test record missing')
    paths = ['paper/arr/main_en.tex', 'paper/arr/中文对应稿.md', 'paper/arr/RESPONSIBLE_NLP_DRAFT.md',
             'paper/arr/translation-audit.json', 'paper/arr/native-compile-diagnostic.json',
             'handoff/remaining-registered-experiments/completed.json', 'handoff/tests-final-registered-experiments.log',
             'handoff/REPRODUCE_WINDOWS.md', 'paper/READING_GUIDE.md',
             'handoff/JEV_API_COMPARISON_PROTOCOL.md', 'handoff/tests-jev-budget-safety.log',
             'paper/figures/fresh_clinc_expansion-evidence.json',
             'paper/figures/domain_accuracy_runtime-evidence.json']
    paths += [item['path'] for item in evidence['source_files'].values()]
    if v2_path.exists():paths += ['results/caden-multidomain-v2/integrity-audit.json','handoff/tests-caden-v2-working.log']
    source_bundle_path = 'paper/submission-source/paper-source-' + digest[:16] + '.audit.json'
    bundle = read(source_bundle_path)
    require(bundle['english_master_sha256'] == digest and bundle['all_file_hashes_verified']
            and bundle['literal_scan_passed'] and not bundle['submission_ready'], 'Source bundle mismatch')
    require(sha(bundle['archive']) == bundle['archive_sha256'], 'Source bundle changed')
    paths += [source_bundle_path, bundle['archive'], 'handoff/CITATION_REVIEW.md']
    report = {'checked_at': datetime.now().astimezone().isoformat(), 'english_master_sha256': digest,
              'completed_evidence': {'registered_queue_stages': 6, 'fresh_confirmation_questions': 4197,
                'fresh_arms': 15, 'candidate_trace_pairs': 38, 'task_model_blocks': 70,
                'synthetic_shapes': 6, 'synthetic_blocks_per_shape': 5, 'tests_passed': 63,
                'translated_sections': 14, 'tables': translation['table_count'], 'table_projection_lines': translation['projected_table_lines_verified'], 'jev_model': jev['model'],
                'jev_final_questions': 7796, 'jev_paid_requests': 7800,
                'jev_estimated_spent_usd': jev['budget']['estimated_spent_usd'], 'api_budget_safety_tests_passed': 5},
              'files': [{'path': path, 'sha256': sha(path)} for path in paths],
              'remaining': [
                {'gate': 'Paper PDF and ARR page limit', 'state': 'unverified',
                 'reason': diagnostic['log'].strip()},
                {'gate': 'Submission package anonymity and human author review', 'state': 'not_certified',
                 'reason': 'Local integrity snapshot is not a cleared anonymous upload; authors must verify code, citations, disclosures and submission declarations.'}],
              'submission_ready': False,
              'scope': 'Checks completeness and current file hashes against already audited experiment summaries. Does not rerun GPU experiments, certify all historical compute, publish materials or validate a paper PDF.'}
    (ROOT / 'handoff/FINAL_RESEARCH_AUDIT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'files'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
