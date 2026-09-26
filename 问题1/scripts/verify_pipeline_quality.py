"""Independently recount the full controlled experiment and its denominators."""
from collections import Counter
from pathlib import Path
import numpy as np
from common import read, write, digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT/'data/quality_evaluation'
    result = read(out/'results.json')
    protocol = read(out/'protocol.json')
    assert result['status'] == 'passed'
    assert result['protocol_sha256'] == digest(out/'protocol.json')
    assert result['human_records_used'] is False
    assert result['semantic_accuracy_measured'] is False
    for name, sha in protocol['scripts'].items():
        assert digest(ROOT/'scripts'/name) == sha, name
    source = {s['sample_id']:s for s in read(ROOT/'data/inputs.json')['samples']}
    assert len(source) == len(result['samples']) == 100
    assert {r['sample_id'] for r in result['samples']} == set(source)
    expected = {(sid, name) for sid in source for name in protocol['variants']}
    assert len(result['experiments']) == len(expected) == 300
    assert {(r['sample_id'],r['variant']) for r in result['experiments']} == expected
    totals = {name:Counter() for name in protocol['variants']}
    errors = {name:[] for name in protocol['variants']}
    in_added_silence = 0
    checked_words = 0
    for row in result['experiments']:
        sid, name = row['sample_id'], row['variant']
        original = read(ROOT/'data/local_evidence_corroborated'/(sid+'.json'))
        meta = read(ROOT/'data/features'/(sid+'.json'))
        assert len(row['words']) == len(original['words'])
        shift = 1. if name == 'leading_silence_1s' else 0.
        counts = Counter(reference_candidates=original['automatic_eligible_words'])
        for old, new in zip(original['words'], row['words']):
            assert old['word_index'] == new['word_index']
            b, a = old['automatic_eligible'], new['candidate']
            counts['candidates'] += a
            counts['retained'] += bool(a and b)
            counts['lost'] += bool(b and not a)
            counts['added'] += bool(a and not b)
            if a:
                start, end = new['interval_s']
                assert np.isfinite([start,end]).all() and source[sid]['audio']['start_s'] <= start < end
                assert end <= source[sid]['audio']['start_s']+meta['resampled_samples']/16000+shift+1e-8
                assert not source[sid]['zero_audio']
                if shift and end <= source[sid]['audio']['start_s']+shift:
                    in_added_silence += 1
            else:
                assert new['interval_s'] is None
            if a and b:
                error = max(abs(np.asarray(new['interval_s'])-shift-np.asarray(old['interval_s'])))
                assert abs(new['shift_corrected_boundary_residual_s']-error) < 1e-12
                errors[name].append(float(error))
            else:
                assert new['shift_corrected_boundary_residual_s'] is None
            if name == 'identity':
                assert a == b and old['interval_s'] == new['interval_s'] and old['status'] == new['status']
            checked_words += 1
        for key in ['reference_candidates','candidates','retained','lost','added']:
            assert counts[key] == row[key], (sid,name,key)
        totals[name].update(counts)
    for name, counts in totals.items():
        v = result['transformation_summary'][name]
        for key, count in counts.items():
            assert v[key] == count
        assert v['boundary_residual_s']['n'] == len(errors[name])
        assert abs(v['boundary_residual_s']['p95']-np.quantile(errors[name],.95)) < 1e-12
        assert v['retained_within_20ms'] == sum(e<=.020000001 for e in errors[name])
        assert v['retained_within_100ms'] == sum(e<=.100000001 for e in errors[name])
    summary = read(out/'summary.json')
    assert summary['results_sha256'] == digest(out/'results.json')
    assert sum(summary['unresolved_direct_reasons'].values()) == 590
    write(out/'verification.json', {'status':'passed', 'samples':100,'experiments':300,
        'word_records_checked':checked_words, 'identity_reproduced':True,
        'counts_and_residual_denominators_verified':True,
        'candidates_fully_inside_added_silence':in_added_silence,
        'protocol_sha256':digest(out/'protocol.json'), 'results_sha256':digest(out/'results.json'),
        'summary_sha256':digest(out/'summary.json'),
        'verification_script_sha256':digest(Path(__file__)),
        'scope':'Consistency and controlled-transform measurements; not absolute alignment accuracy.'})
    print('Verified',checked_words,'word records across 300 controlled runs; silence-only candidates:',in_added_silence)


if __name__ == '__main__':
    main()
