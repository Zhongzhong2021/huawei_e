"""Rebuild a real trimodal example using only the selected delivery pipeline."""
import argparse
import io
import textwrap
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from common import read, write, digest, run
from delivery_reader import Dataset
from temporal_support import interval_relation

ROOT = Path(__file__).resolve().parents[1]


def show_features(ax, values, mask, title):
    x = np.asarray(values, dtype=float)
    scale = np.maximum(np.max(np.abs(x), axis=0), 1e-12)
    image = np.ma.masked_where(~np.asarray(mask, dtype=bool), x / scale)
    cmap = plt.get_cmap('coolwarm').copy(); cmap.set_bad('#808080')
    ax.imshow(image.T, aspect='auto', interpolation='nearest', cmap=cmap, vmin=-1, vmax=1)
    ax.set_title(title); ax.set_xlabel('Native selected row order'); ax.set_ylabel('Feature dimension')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-id')
    parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    dataset = Dataset(ROOT)
    if args.sample_id is None:
        for item in sorted(dataset.records.values(), key=lambda r:r['sha256']):
            candidate = dataset.sample(item['sample_id'])
            if all(w['automatic_eligible'] for w in candidate.automatic['words']):
                args.sample_id = item['sample_id']
                break
        if args.sample_id is None:
            raise ValueError('No sample has full candidate coverage for the illustration')
    sample = dataset.sample(args.sample_id)
    if not all(w['automatic_eligible'] for w in sample.automatic['words']):
        raise ValueError('This full-text example requires a candidate for every original word')
    inputs = read(ROOT / 'data/inputs.json')
    source_root = args.source if args.source is not None else Path(inputs['source_root'])
    source = source_root / sample.source['path']
    if digest(source) != sample.source['sha256']:
        raise ValueError('Original video digest mismatch')
    words = sample.automatic['words']
    interval = [words[0]['interval_s'][0], words[-1]['interval_s'][1]]
    relation = interval_relation(sample.arrays, sample.metadata, sample.source, interval,
                                 'automatic_candidate', range(len(words)))
    out = ROOT / 'data/figures'; out.mkdir(exist_ok=True)
    pcm = np.frombuffer(run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(source),
        '-map', '0:a:0', '-f', 'f32le', '-']), '<f4').reshape(-1, sample.source['audio']['channels'])
    rate = int(sample.source['audio']['sample_rate'])
    assert len(pcm) == sample.source['pcm_samples']
    figure = plt.figure(figsize=(15, 10), constrained_layout=True)
    grid = figure.add_gridspec(3, 3, height_ratios=[1.1, 1.2, 1.0])
    ax = figure.add_subplot(grid[0, :]); step = max(1, len(pcm) // 16000)
    ax.plot(sample.source['audio']['start_s'] + np.arange(0, len(pcm), step) / rate,
            pcm.mean(1)[::step], linewidth=.45)
    ax.axvspan(*interval, color='green', alpha=.15, label='Automatic text envelope (candidate)')
    ax.set_xlabel('Original source time (seconds)'); ax.set_ylabel('PCM amplitude'); ax.legend(loc='upper right')
    ax.set_title('\n'.join(textwrap.wrap(sample.metadata['original_text'], 120)))
    for column, key, rows, mask in [
        (0, 'text', relation['text_rows'], np.ones_like(sample.arrays['text'], dtype=bool)),
        (1, 'audio', relation['audio']['rows'], sample.arrays['audio_dim_mask']),
        (2, 'scene', relation['scene']['rows'], np.ones_like(sample.arrays['scene'], dtype=bool))]:
        show_features(figure.add_subplot(grid[1, column]), sample.arrays[key][rows], mask[rows],
                      f'{key}: {len(rows)} rows x {sample.arrays[key].shape[1]} dimensions')
    scene_rows = relation['scene']['rows']
    selected = scene_rows[np.linspace(0, len(scene_rows) - 1, 3, dtype=int)]
    frames = []
    for column, row in enumerate(selected):
        index = int(sample.arrays['video_frame_indices'][row])
        blob = run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(source), '-vf',
            f'select=eq(n\\,{index})', '-frames:v', '1', '-f', 'image2pipe', '-vcodec', 'png', '-'])
        image_path = out / f'delivery-example-frame-{index}.png'; image_path.write_bytes(blob)
        ax = figure.add_subplot(grid[2, column]); ax.imshow(Image.open(io.BytesIO(blob))); ax.axis('off')
        pts = float(sample.arrays['video_times'][row])
        ax.set_title(f'Original frame {index}, PTS {pts:.3f}s; scene row {row}')
        frames.append({'scene_row': int(row), 'source_frame_index': index, 'pts_s': pts,
                       'image': str(image_path.relative_to(ROOT)), 'image_sha256': digest(image_path)})
    figure.suptitle('Selected delivery policy: original text, source audio, real frames and native features\n'
                     'Display scaling only; gray = undefined dimension; no human times or speaker identity inferred.', fontsize=13)
    figure.savefig(out / 'delivery_example.png', dpi=140)
    figure.savefig(out / 'delivery_example.pdf')
    plt.close(figure)
    write(out / 'delivery_example.json', {'sample_id': args.sample_id,
        'default_selection_rule': 'Smallest source SHA256 among samples with full baseline candidate coverage; no human records used.',
        'source_sha256': sample.source['sha256'], 'original_text': sample.metadata['original_text'],
        'automatic_group_envelope_s': interval, 'word_intervals_s': [w['interval_s'] for w in words],
        'group_envelope_includes_inter_word_pauses': True, 'word_boundaries_verified': False,
        'text_rows': relation['text_rows'].tolist(), 'audio_rows': relation['audio']['rows'].tolist(),
        'audio_overlap_seconds': relation['audio']['row_overlap_seconds'].tolist(),
        'scene_rows': scene_rows.tolist(), 'scene_overlap_seconds': relation['scene']['row_overlap_seconds'].tolist(),
        'frames': frames, 'human_data_used_for_generation': False,
        'script_sha256': digest(Path(__file__))})
    print('Built delivery-policy example:', args.sample_id, interval)


if __name__ == '__main__':
    main()
