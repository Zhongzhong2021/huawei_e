"""Full-data independent formula checks for the physical temporal index."""
import hashlib
import json
from fractions import Fraction
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = ROOT / 'data/temporal_index'
    manifest = json.loads((out / 'manifest.json').read_text())
    inputs = json.loads((ROOT / 'data/inputs.json').read_text())
    records = {r['sample_id']: r for r in manifest['samples']}
    assert set(records) == {r['sample_id'] for r in inputs['samples']} and len(records) == 100
    for name, digest in manifest['scripts'].items():
        assert sha(ROOT / 'scripts' / name) == digest
    for item in inputs['samples']:
        sid = item['sample_id']
        record = records[sid]
        assert sha(Path(inputs['source_root']) / item['path']) == record['source_sha256']
        for key, path in [('native_features_sha256', ROOT / 'data/features' / (sid + '.npz')),
                          ('native_metadata_sha256', ROOT / 'data/features' / (sid + '.json')),
                          ('index_sha256', out / (sid + '.npz'))]:
            assert sha(path) == record[key]
        with np.load(ROOT / 'data/features' / (sid + '.npz')) as native, np.load(out / (sid + '.npz')) as index:
            tick = float(Fraction(item['video']['time_base']))
            frame_starts = np.asarray(item['video']['pts_ticks']) * tick
            frame_ends = frame_starts + np.asarray(item['video']['lengths']) * tick
            np.testing.assert_array_equal(index['target_intervals'], native['video_cells'])
            assert all(np.isfinite(index[k]).all() for k in index.files)
            assert not index['audio_values'][~index['audio_dimension_mask']].any()
            assert len(index['audio_indptr']) == len(index['target_intervals']) + 1
            for i, (lo, hi) in enumerate(index['target_intervals']):
                start, stop = index['audio_indptr'][i:i+2]
                rows, weights = index['audio_rows'][start:stop], index['audio_overlap_seconds'][start:stop]
                original_start = item['audio']['start_s']
                original_end = original_start + item['pcm_samples'] / int(item['audio']['sample_rate'])
                assert abs(original_end - item['audio']['end_s']) < 1e-9
                expected = np.maximum(0, np.minimum(native['audio_cells'][:, 1], min(hi, original_end))
                                      - np.maximum(native['audio_cells'][:, 0], max(lo, original_start)))
                np.testing.assert_array_equal(rows, np.flatnonzero(expected > 0))
                np.testing.assert_allclose(weights, expected[rows], atol=1e-12, rtol=0)
                np.testing.assert_allclose(index['audio_observed_seconds'][i], expected.sum(), atol=1e-12, rtol=0)
                valid = native['audio_dim_mask'][rows] & (not item['zero_audio'])
                denominator = (weights[:, None] * valid).sum(0)
                expected_values = np.divide((native['audio'][rows] * weights[:, None] * valid).sum(0), denominator, out=np.zeros(17), where=denominator > 0)
                np.testing.assert_allclose(index['audio_values'][i], expected_values, rtol=1e-6, atol=1e-6)
                np.testing.assert_allclose(index['audio_valid_seconds'][i], denominator, atol=1e-12, rtol=0)
                assert index['scene_observed_seconds'][i] <= hi - lo + 1e-9
                # Independent endpoint partition: a piece is observed iff its
                # midpoint belongs to at least one decoded frame support.
                endpoints = np.unique(np.r_[lo, hi, frame_starts[(frame_starts > lo) & (frame_starts < hi)], frame_ends[(frame_ends > lo) & (frame_ends < hi)]])
                mids = (endpoints[:-1] + endpoints[1:]) / 2
                covered = ((mids[:, None] >= frame_starts) & (mids[:, None] < frame_ends)).any(1)
                actual_scene_seconds = float(np.diff(endpoints)[covered].sum())
                np.testing.assert_allclose(index['scene_observed_seconds'][i], actual_scene_seconds, rtol=0, atol=1e-9)
                fstart, fstop = index['face_indptr'][i:i+2]
                np.testing.assert_array_equal(index['face_event_rows'][fstart:fstop], np.flatnonzero(native['face_frame_rows'] == i) if index['scene_observed_seconds'][i] > 0 else [])
            assert record['original_text_words_retained'] == len(native['text'])
            if item['zero_audio']:
                assert not index['audio_dimension_mask'].any()
    result = {'passed': True, 'samples': 100, 'time_nodes': manifest['total_time_nodes'],
              'source_and_native_hashes_checked': True, 'original_text_words': sum(r['original_text_words_retained'] for r in records.values()),
              'scope': 'Independent audio intersection/aggregation formulas, source identity, native frame/face correspondence and masks; not semantic alignment validation.'}
    (out / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
