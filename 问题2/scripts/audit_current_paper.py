"""Check integrated manuscript tables against frozen evidence, never select a model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recorded_path(value):
    if os.name == 'nt' and value.startswith('/mnt/'):
        return Path(value[5].upper()+':/'+value[7:])
    if os.name == 'nt' and value.startswith('/home/'):
        return Path('//wsl.localhost/Ubuntu-22.04'+value)
    if os.name != 'nt' and re.match(r'^[A-Za-z]:', value):
        return Path('/mnt/'+value[0].lower()+'/'+value[3:].replace('\\', '/'))
    return Path(value)


def main(root, docs):
    report = root/'audit/manuscript_checks.json'
    assert not report.exists(), 'Preserve completed audits'
    evidence = read(root/'analysis/current_evidence.json')
    checks = read(root/'audit/evidence_checks.json')
    provenance = read(root/'reports/current_provenance.json')
    blocks = read(root/'reports/current_content.json')
    for path, expected in checks['source_hashes'].items():
        assert digest(recorded_path(path)) == expected, path
    for group in ['source_documents', 'source_studies']:
        for record in provenance[group].values():
            assert digest(recorded_path(record['path'])) == record['sha256']
    assert digest(root/'analysis/current_evidence.json') == provenance['evidence_sha256']
    scripts = Path(__file__).parent
    assert digest(scripts/'paper_current.py') == provenance['script_sha256']
    assert digest(scripts/'build_paper.py') == provenance['word_builder_sha256']
    assert digest(scripts/'prepare_current_paper.py') == checks['script_sha256']
    assert provenance['current_model_sha256'] == evidence['frozen']['checkpoint_sha256'] == checks['model_sha256']
    for path, record in provenance['copied_figure_sources'].items():
        assert digest(root/path) == record['sha256'] == digest(recorded_path(record['source']))
    for record in read(root/'figures/current_figure_manifest.json'):
        assert record['source_sha256'] == digest(root/record['source'])
        assert record['script_sha256'] == checks['script_sha256']
        assert record['model_sha256'] == checks['model_sha256']
        assert all((root/'figures'/(record['id']+'.'+ext)).exists() for ext in record['formats'])
    tables = {b['key']: b for b in blocks if b['type'] == 'table' and b.get('key')}
    mods = {'text':'文本', 'audio':'语音', 'vision':'视觉'}
    for key, source_name, columns in [
        ('current_main', 'main_missing', ['macro_f1','delta_f1','mae','delta_mae']),
        ('current_shapes', 'shape_comparison', ['single_f1','multi_f1','delta_f1','delta_mae'])]:
        records = evidence['tables'][source_name]
        assert len(tables[key]['rows']) == len(records) == 9
        for row, record in zip(tables[key]['rows'], records):
            assert row[:2] == [mods[record['modality']], str(record['rate'])+'%']
            for cell, column in zip(row[2:], columns):
                assert cell == format(record[column], '+.4f' if column.startswith('delta') else '.4f')
    cases = evidence['tables']['cases']
    assert len(tables['current_cases']['rows']) == len(cases) == 6
    for row, record in zip(tables['current_cases']['rows'], cases):
        assert row == [record['sample_id'], '错误最大MAE' if record['selection_rule'].startswith('wrong') else '正确中位MAE',
                       f"{record['true_class']}→{record['predicted_class']}", f"{record['true_intensity']:.4f}", f"{record['predicted_intensity']:.4f}"]
    special = evidence['tables']['attachment3']
    assert len(tables['current_special']['rows']) == len(special) == 30
    for row, record in zip(tables['current_special']['rows'], special):
        assert row == [record['sample_id'].split('::')[0].replace('.pkl',''), ['负向','中性','正向'][int(record['class_id'])],
                       f"{float(record['intensity']):.4f}", f"{max(float(record['prob_'+c]) for c in ['negative','neutral','positive']):.4f}"]
    r5 = read(docs/'round5_snapshot/study/development.json')['entries'][0]
    assert tables['defer']['rows'] == [['延后1轮−立即', *[f"{r5['guard']['deltas'][k]:+.6f}" for k in ['score','clean_f1','clean_mae','accuracy','positive_f1']], str(r5['fold_improvements'])+'/3']]
    r6 = read(docs/'round6/study/confirmed.json')
    assert tables['blend']['rows'] == [[str(rec['seed']), *[f"{rec['deltas'][k]:+.6f}" for k in ['score','clean_f1','clean_mae','neutral_f1','positive_f1']], '通过' if rec['passed'] else '未通过'] for rec in r6['pairs']+[{'seed':'均值', **r6['mean_guard']}]]
    # Independently recompute the headline table from all three recorded seeds.
    main_table = next(b for b in blocks if b['_origin']=='r4' and b['_source_index']==31)
    getters = [lambda m:m['clean']['accuracy'], lambda m:m['clean']['macro_f1'], lambda m:m['clean']['mae'], lambda m:m['clean']['pearson']]
    getters += [(lambda m, c=c:m['clean']['class_f1'][c]) for c in range(3)]
    getters += [(lambda m, k=k:m['selection'][k]) for k in ['mean_missing_macro_f1','mean_missing_mae','score']]
    for row, getter in zip(main_table['rows'], getters):
        values = [[getter(r[role]) for r in evidence['confirmation']['records']] for role in ['reference','metrics']]
        assert row[1:3] == [f'{statistics.mean(v):.4f} ± {statistics.stdev(v):.4f}' for v in values]
        assert row[3] == f'{statistics.mean(values[1])-statistics.mean(values[0]):+.4f}'
    # Historical table values stay unchanged; only explicitly named labels differ.
    originals = {k:read(recorded_path(v['path'])) for k,v in provenance['source_documents'].items()}
    for b in blocks:
        if b['type'] == 'table' and b['_origin'] in originals:
            prior = originals[b['_origin']][b['_source_index']]
            if (b['_origin'],b['_source_index']) not in [('r3',48),('r3',49),('r3',65)]:
                assert b['rows'] == prior['rows']
    for typ, prefix in [('equation',None),('figure','图'),('table','表')]:
        subset = [b for b in blocks if b['type']==typ and (typ != 'table' or b['caption'].startswith('表'))]
        numbers = [b['number'] if typ=='equation' else int(re.match(prefix+r'(\d+)',b['caption'])[1]) for b in subset]
        assert numbers == list(range(1,len(numbers)+1))
    result = {'current_model_sha256':checks['model_sha256'], 'model_changed':False,
        'original_source_hashes_rechecked':len(checks['source_hashes']),
        'current_tables_verified':['current_main','current_shapes','current_cases','current_special','defer','blend','three_seed_headline'],
        'historical_numeric_tables_preserved':True, 'sequential_numbering_verified':True,
        'figure_sources_verified':True, 'manuscript_sha256':digest(root/'reports/current_content.json'),
        'auditor_sha256':digest(Path(__file__)), 'note':'Publication audit; does not rerun model selection or infer new predictive improvement.'}
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--docs', type=Path, required=True)
    a = p.parse_args()
    main(a.root, a.docs)
