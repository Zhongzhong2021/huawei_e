"""Summarize actual native data through the portable reader, without inference."""
import argparse
import csv
from collections import Counter
from pathlib import Path
from common import write, digest
from delivery_reader import Dataset
from temporal_support import source_video_support


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    dataset = Dataset(args.root)
    rows, statuses = [], Counter()
    for sid in dataset.records:
        s = dataset.sample(sid)
        a, source = s.arrays, s.source
        support = source_video_support(source['video'])
        count = sum(w['automatic_eligible'] for w in s.automatic['words'])
        statuses[s.automatic['sample_status']] += 1
        rows.append(dict(sample_id=sid, modalities='text/audio/scene/face-events',
            audio_observed_s=source['pcm_samples'] / int(source['audio']['sample_rate']),
            audio_signal_usable=not source['zero_audio'],
            video_observed_s=float((support[:, 1] - support[:, 0]).sum()),
            text_shape=f"{len(a['text'])}x768", audio_shape=f"{len(a['audio'])}x17",
            scene_shape=f"{len(a['scene'])}x512", face_shape=f"{len(a['faces'])}x52",
            automatic_candidate_words=count, unresolved_words=len(a['text']) - count,
            baseline_candidate_words=s.baseline_automatic['automatic_eligible_words'],
            locally_restored_words=s.automatic['new_local_candidate_words'],
            sample_content_status=s.automatic['sample_status'],
            granularity='native sequences; physical AV cells; selective word candidates',
            feature_file=f'data/features/{sid}.npz', index_file=f'data/temporal_index/{sid}.npz'))
    output = args.root / 'data/delivery_summary'
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'all_100.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    table = ['# 全量交付汇总（自动候选与未确定关系）', '',
        '全部样本保留文本、声音、画面及人脸事件。声音时长来自实际PCM，视频时长为源帧支持并集，不使用容器时长。零音轨有物理时长但没有可用语音信号。形状为实际序列长度×维度；人脸长度按检测事件计。', '',
        '对齐粒度统一为原生变长序列、音视频物理时间单元与选择性词级候选；未确定词不分配替代时间。“候选/未定”不表示正确/错误。详细状态、路径及未四舍五入秒数见同目录CSV。', '',
        '| 样本 | 声音秒 | 视频秒 | 文本形状 | 声音形状 | 画面形状 | 人脸形状 | 候选/未定 | 零音轨 |',
        '|---|---:|---:|---|---|---|---|---:|---|']
    for r in rows:
        table.append(f"| {r['sample_id']} | {r['audio_observed_s']:.6f} | {r['video_observed_s']:.6f} | {r['text_shape']} | {r['audio_shape']} | {r['scene_shape']} | {r['face_shape']} | {r['automatic_candidate_words']}/{r['unresolved_words']} | {'是' if not r['audio_signal_usable'] else '否'} |")
    (output / 'all_100.md').write_text('\n'.join(table) + '\n')
    write(output / 'verification.json', dict(samples=len(rows),
        automatic_candidate_words=sum(r['automatic_candidate_words'] for r in rows),
        unresolved_words=sum(r['unresolved_words'] for r in rows),
        sample_status_counts=dict(statuses), reader_identity_and_hash_checks_passed=True,
        quality_accuracy_verified=False, human_data_read=False,
        inputs_sha256=digest(args.root / 'data/inputs.json'),
        scripts={p: digest(args.root / 'scripts' / p) for p in
                 ['q1_reader.py', 'delivery_reader.py', 'local_evidence.py', 'build_delivery_summary.py', 'temporal_support.py']}))
    print(f'Summarized {len(rows)} samples; candidate counts are not accuracy.')


if __name__ == '__main__':
    main()
