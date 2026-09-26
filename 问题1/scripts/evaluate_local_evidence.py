"""Post-generation development evaluation and cross-video lexical stress control."""
from collections import Counter
import argparse
from pathlib import Path
import numpy as np
from common import read, write, digest
from feature_io import load
from local_evidence import local_candidates, corroborated_local_candidates
from selective_alignment import align_sample
from temporal_support import interval_relation

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corroborated', action='store_true')
    args = parser.parse_args()
    decide = corroborated_local_candidates if args.corroborated else local_candidates
    out = ROOT / ('data/local_evidence_corroborated' if args.corroborated else 'data/local_evidence')
    manifest = read(out / 'manifest.json')
    config = read(ROOT / 'config/automatic_alignment.json')
    inputs = read(ROOT / 'data/inputs.json')['samples']
    baseline, metadata, changed = {}, {}, []
    for source in inputs:
        sid = source['sample_id']
        arrays, meta = load(ROOT / 'data/features', sid)
        metadata[sid] = meta
        original = read(ROOT / 'data/automatic_alignment' / (sid + '.json'))
        baseline[sid] = original
        new = read(out / (sid + '.json'))
        recalculated = decide(original, config)
        assert new['words'] == recalculated['words']
        assert new['baseline_result_sha256'] == digest(ROOT / 'data/automatic_alignment' / (sid + '.json'))
        for old_word, word in zip(original['words'], new['words']):
            if old_word['automatic_eligible']:
                assert word == old_word
            if word['status'] == 'local_coherent_candidate':
                r = interval_relation(arrays, meta, source, word['interval_s'],
                                      'automatic_candidate', [word['word_index']])
                assert r['audio']['usable']
                changed.append({'sample_id': sid, 'word_index': word['word_index'],
                                'text': word['text'], 'interval_s': word['interval_s']})
    # Read references only after generated outputs have been independently recomputed.
    groups = {}
    reference = read(ROOT / 'data/independent_content_review.json')
    for item in reference['samples']:
        sid = item['sample_id']; new = read(out / (sid + '.json'))
        group = groups.setdefault(item['content_relation'], {'samples': 0, 'words': 0,
            'baseline_candidates': 0, 'new_candidates': 0})
        group['samples'] += 1; group['words'] += len(new['words'])
        group['baseline_candidates'] += baseline[sid]['automatic_eligible_words']
        group['new_candidates'] += new['automatic_eligible_words']
    previous_mismatch = []
    for item in read(ROOT / 'data/automatic_evaluation.json')['diagnostic_existing_human_reports']:
        if item['human_interpretation'] == 'human_reported_full_audio_text_mismatch':
            sid = item['sample_id']
            previous_mismatch.append({'sample_id': sid,
                'candidate_words': read(out / (sid + '.json'))['automatic_eligible_words']})
    controls = []
    nonzero = [s for s in inputs if not s['zero_audio']]
    for audio in nonzero:
        sid = audio['sample_id']
        for text in nonzero:
            if text['video_id'] == audio['video_id']:
                continue
            target = text['sample_id']; words = metadata[target]['words']
            test = align_sample(words, baseline[sid]['recognition'], np.zeros((len(words), 2)), config)
            test['recognition'] = baseline[sid]['recognition']
            # Deliberately grant perfect boundary agreement to eligible lexical matches.
            # This avoids trivially rejecting swapped references via zero dummy timestamps.
            for word in test['words']:
                exact = (word['diagnostic_interval_s'] is not None
                         and all(p['exact'] for p in word['recognition_options']))
                word['boundary_disagreement_s'] = 0 if exact else word['boundary_disagreement_s']
                word['automatic_eligible'] = bool(exact and test['sample_status'] == 'supported')
                word['interval_s'] = word['diagnostic_interval_s'] if word['automatic_eligible'] else None
                if word['automatic_eligible']:
                    word['status'] = 'automatic_candidate'
            test['automatic_eligible_words'] = sum(w['automatic_eligible'] for w in test['words'])
            local = decide(test, config)
            controls.append({'audio_sample_id': sid, 'reference_sample_id': target,
                'baseline_lexical_candidates': test['automatic_eligible_words'],
                'local_added_candidates': local['new_local_candidate_words']})
        print('Controlled audio', sid, 'pairs so far', len(controls), flush=True)
    write(out / 'evaluation.json', {'status': 'development_evaluation_not_independent_accuracy',
        'all_100_recomputed': True, 'existing_candidates_unchanged': True,
        'new_word_relations_checked': len(changed), 'changed_words': changed,
        'content_reference_groups': groups, 'previous_full_audio_mismatch': previous_mismatch,
        'cross_video_control': {'pairs': len(controls),
            'baseline_pairs_with_candidates': sum(r['baseline_lexical_candidates'] > 0 for r in controls),
            'pairs_with_local_additions': sum(r['local_added_candidates'] > 0 for r in controls),
            'cases': controls,
            'meaning': 'Synthetic cross-video references with oracle boundary agreement; tests lexical safeguards only, not natural error rate.'},
        'generation_manifest_sha256': digest(out / 'manifest.json'),
        'reference_sha256': digest(ROOT / 'data/independent_content_review.json'),
        'word_accuracy_measured': False, 'human_times_used': False})
    print('Finished evaluation:', len(changed), 'additions;', len(controls), 'stress pairs')


if __name__ == '__main__':
    main()
