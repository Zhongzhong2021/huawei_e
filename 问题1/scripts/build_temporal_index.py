"""Materialize a compact native-time audio/visual index for all 100 samples.

Each target is a sampled scene feature's existing temporal cell. This is a
physical organization layer; text-to-speech candidates remain separate.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from temporal_support import interval_relation
from feature_io import load

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = ROOT / 'data/temporal_index'
    out.mkdir(exist_ok=False)
    inputs = json.loads((ROOT / 'data/inputs.json').read_text())
    summary = []
    for item in inputs['samples']:
        sid = item['sample_id']
        arrays, meta = load(ROOT / 'data/features', sid)
        source = Path(inputs['source_root']) / item['path']
        assert sha(source) == item['sha256']
        targets = arrays['video_cells']
        pointer, audio_rows, audio_weights = [0], [], []
        face_pointer, face_rows = [0], []
        audio_values, audio_valid, audio_seconds, scene_seconds = [], [], [], []
        observed_audio_seconds, pitch_dilution = [], []
        for target in targets:
            relation = interval_relation(arrays, meta, item, target, 'physical_time_window')
            audio, scene = relation['audio'], relation['scene']
            audio_rows.extend(audio['rows'].tolist())
            audio_weights.extend(audio['row_overlap_seconds'].tolist())
            pointer.append(len(audio_rows))
            face_rows.extend(e['row'] for e in relation['face_events'])
            face_pointer.append(len(face_rows))
            audio_values.append(audio['values'])
            audio_valid.append(audio['dimension_mask'])
            audio_seconds.append(audio['valid_seconds_per_dimension'])
            observed_audio_seconds.append(audio['observed_seconds'])
            scene_seconds.append(scene['observed_seconds'])
            # Controlled algebraic ablation: zero-filled averaging vs valid-only.
            if audio['dimension_mask'][-2] and audio['observed_seconds'] > 0:
                naive = float(np.dot(arrays['audio'][audio['rows'], -2], audio['row_overlap_seconds']) / audio['observed_seconds'])
                pitch_dilution.append(float(audio['values'][-2] - naive))
        np.savez_compressed(out / (sid + '.npz'),
            target_intervals=targets, source_scene_rows=np.arange(len(targets), dtype=np.int64),
            audio_indptr=np.array(pointer, np.int64), audio_rows=np.array(audio_rows, np.int64),
            audio_overlap_seconds=np.array(audio_weights, np.float64),
            audio_values=np.asarray(audio_values, np.float32).reshape(-1, 17),
            audio_dimension_mask=np.asarray(audio_valid, bool).reshape(-1, 17),
            audio_valid_seconds=np.asarray(audio_seconds, np.float64).reshape(-1, 17),
            audio_observed_seconds=np.asarray(observed_audio_seconds, np.float64),
            scene_observed_seconds=np.asarray(scene_seconds, np.float64),
            face_indptr=np.asarray(face_pointer, np.int64), face_event_rows=np.asarray(face_rows, np.int64))
        record = {'sample_id': sid, 'source_sha256': item['sha256'],
                  'native_features_sha256': sha(ROOT / 'data/features' / (sid + '.npz')),
                  'native_metadata_sha256': sha(ROOT / 'data/features' / (sid + '.json')),
                  'index_sha256': sha(out / (sid + '.npz')), 'time_nodes': len(targets),
                  'audio_edges': len(audio_rows), 'face_events_linked': len(face_rows),
                  'original_text_words_retained': len(arrays['text']), 'zero_audio': item['zero_audio'],
                  'text_time_inferred': False, 'source_scene_semantics': 'Nearest sampled frame representation over its native cell, restricted to the union of actual decoded frame supports.',
                  'audio_usable_nodes': int(np.any(audio_valid, axis=1).sum()),
                  'video_gap_seconds_excluded': float(np.diff(targets, axis=1).sum() - sum(scene_seconds)),
                  'voiced_nodes': len(pitch_dilution),
                  'nodes_with_pitch_zero_fill_dilution': sum(d > 1e-6 for d in pitch_dilution),
                  'max_pitch_zero_fill_dilution_hz': max(pitch_dilution, default=0),
                  'semantic_acceptance': 'not_applicable_to_physical_index', 'human_data_used': False}
        (out / (sid + '.json')).write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        summary.append(record)
    manifest = {'schema': 'q1-native-time-index-v1', 'samples': summary,
                'scripts': {n: sha(ROOT / 'scripts' / n) for n in ['temporal_support.py', 'build_temporal_index.py']},
                'total_samples': len(summary), 'total_time_nodes': sum(r['time_nodes'] for r in summary),
                'total_audio_edges': sum(r['audio_edges'] for r in summary),
                'total_video_gap_seconds_excluded': sum(r['video_gap_seconds_excluded'] for r in summary),
                'nodes_with_pitch_zero_fill_dilution': sum(r['nodes_with_pitch_zero_fill_dilution'] for r in summary),
                'claim': 'Time support and per-dimension validity are enforced; no emotion-accuracy or semantic-alignment gain is inferred.'}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k not in ['samples', 'scripts']}))


if __name__ == '__main__':
    main()
