"""Time-support-aware aggregation without inventing semantic correspondence.

Native features remain unchanged. Missing dimensions have their own denominator;
scene support excludes gaps in decoded source video. Face events keep identity
unknown and are returned individually, never averaged across people.
"""
from fractions import Fraction
import numpy as np


def intervals(values):
    result = np.asarray(values, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(result).all() or np.any(result[:, 1] <= result[:, 0]):
        raise ValueError('Intervals must be finite with positive duration')
    return result


def union_intervals(values):
    rows = intervals(values)
    merged = []
    for a, b in rows[np.argsort(rows[:, 0], kind='stable')]:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], float(b))
        else:
            merged.append([float(a), float(b)])
    return np.asarray(merged, dtype=np.float64).reshape(-1, 2)


def source_video_support(video):
    scale = float(Fraction(video['time_base']))
    starts = np.asarray(video['pts_ticks'], dtype=np.float64) * scale
    ends = starts + np.asarray(video['lengths'], dtype=np.float64) * scale
    return union_intervals(np.column_stack([starts, ends]))


def overlap_weights(target, cells, support=None):
    target = intervals([target])[0]
    cells = intervals(cells)
    if len(cells) > 1 and np.any(cells[1:, 0] < cells[:-1, 1] - 1e-9):
        raise ValueError('Feature cells overlap or are not ordered')
    left, right = np.maximum(cells[:, 0], target[0]), np.minimum(cells[:, 1], target[1])
    if support is None:
        return np.maximum(0, right - left)
    weight = np.zeros(len(cells), dtype=np.float64)
    for a, b in union_intervals(support):
        weight += np.maximum(0, np.minimum(right, b) - np.maximum(left, a))
    return weight


def aggregate(values, cells, target, valid=None, support=None, usable=True):
    values = np.asarray(values)
    cells = intervals(cells)
    if values.ndim != 2 or len(values) != len(cells) or not np.isfinite(values).all():
        raise ValueError('Expected finite row-major features matching the cells')
    valid = np.ones(values.shape, bool) if valid is None else np.asarray(valid, dtype=bool)
    if valid.shape != values.shape:
        raise ValueError('Dimension mask shape mismatch')
    weights = overlap_weights(target, cells, support)
    effective = weights[:, None] * valid * bool(usable)
    denominator = effective.sum(0)
    mean = np.divide((values * effective).sum(0), denominator,
                     out=np.zeros(values.shape[1], dtype=np.float64), where=denominator > 0)
    indices = np.flatnonzero(weights > 0)
    duration = float(target[1] - target[0])
    return {'values': mean, 'dimension_mask': denominator > 0,
            'valid_seconds_per_dimension': denominator,
            'observed_seconds': float(weights.sum()),
            'observed_fraction': float(weights.sum() / duration),
            'rows': indices, 'row_overlap_seconds': weights[indices],
            'observation_present': bool(len(indices)), 'usable': bool(usable and np.any(denominator > 0))}


def interval_relation(arrays, metadata, input_record, target, relation_kind, word_indices=()):
    allowed = {'automatic_candidate', 'human_reported_group', 'physical_time_window'}
    if relation_kind not in allowed:
        raise ValueError('Explicit relation provenance required')
    if metadata['source_sha256'] != input_record['sha256']:
        raise ValueError('Source mismatch')
    target = intervals([target])[0]
    audio = aggregate(arrays['audio'], arrays['audio_cells'], target,
                      arrays['audio_dim_mask'],
                      support=[[input_record['audio']['start_s'], input_record['audio']['end_s']]],
                      usable=not input_record['zero_audio'])
    scene = aggregate(arrays['scene'], arrays['video_cells'], target,
                      support=source_video_support(input_record['video']))
    selected = np.asarray(list(word_indices), dtype=np.int64)
    if len(selected) and (selected.min() < 0 or selected.max() >= len(arrays['text']) or np.any(np.diff(selected) <= 0)):
        raise ValueError('Word indices must be in range and strictly increasing')
    if relation_kind == 'physical_time_window' and len(selected):
        raise ValueError('A physical window alone cannot establish text correspondence')
    scene_weights = dict(zip(scene['rows'].tolist(), scene['row_overlap_seconds'].tolist()))
    face_rows = [i for i, row in enumerate(arrays['face_frame_rows']) if int(row) in scene_weights]
    return {'sample_id': metadata['sample_id'], 'interval_s': target.tolist(),
            'relation_kind': relation_kind, 'semantic_verified_by_this_function': False,
            'text_rows': selected, 'audio': audio, 'scene': scene,
            'face_events': [{'row': i, 'scene_row': int(arrays['face_frame_rows'][i]),
                             'overlap_seconds': scene_weights[int(arrays['face_frame_rows'][i])]} for i in face_rows],
            'face_identity_or_speaker_inferred': False}
