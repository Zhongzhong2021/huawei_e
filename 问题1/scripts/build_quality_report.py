"""Summarize measured feature availability, failures and controlled stability."""
from collections import Counter
from pathlib import Path
import numpy as np
from common import read, write, digest
from selective_alignment import align_sample
from local_evidence import corroborated_local_candidates

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT/'data/quality_evaluation'
    results = read(out/'results.json')
    protocol = read(out/'protocol.json')
    assert results['status'] == 'passed'
    assert results['protocol_sha256'] == digest(out/'protocol.json')
    for name, sha in protocol['scripts'].items():
        assert digest(ROOT/'scripts'/name) == sha, name
    assert protocol['config_sha256'] == digest(ROOT/'config/automatic_alignment.json')
    cfg = read(ROOT/'config/automatic_alignment.json')
    reasons, sample_totals = Counter(), Counter()
    sensitivity = {'coverage_0.3':0, 'coverage_0.5':0, 'coverage_0.7':0,
                   'without_boundary_guard':0}
    for row in results['samples']:
        sid = row['sample_id']
        current = read(ROOT/'data/local_evidence_corroborated'/(sid+'.json'))
        base = read(ROOT/'data/automatic_alignment'/(sid+'.json'))
        meta = read(ROOT/'data/features'/(sid+'.json'))
        assert row['candidates'] == current['automatic_eligible_words']
        for w in current['words']:
            if w['automatic_eligible']:
                continue
            if current['sample_status'] == 'no_signal':
                reasons['zero_signal'] += 1
            elif not (w['diagnostic_interval_s'] is not None and
                      all(p['stable'] and p['exact'] for p in w['recognition_options'])):
                reasons['no_unique_exact_recognition'] += 1
            elif w['boundary_disagreement_s'] > cfg['boundary_agreement_s']:
                reasons['boundary_disagreement'] += 1
            else:
                reasons['sample_support_only'] += 1
        with np.load(ROOT/'data/features'/(sid+'.npz'), allow_pickle=False) as a:
            baseline = a['word_intervals'].copy()
        for name in sensitivity:
            altered = dict(cfg)
            if name.startswith('coverage_'):
                altered['minimum_informative_coverage'] = float(name.split('_')[1])
            else:
                altered['boundary_agreement_s'] = float('inf')
            trial = align_sample(meta['words'], base['recognition'], baseline, altered,
                                 no_signal=row['zero_audio'])
            trial['recognition'] = base['recognition']
            sensitivity[name] += corroborated_local_candidates(trial, altered)['automatic_eligible_words']
        for key in ['words','candidates','audio_rows','video_rows','no_face_frames',
                    'single_face_frames','multiple_face_frames','face_events','crop_edge_events']:
            sample_totals[key] += row[key]
    assert sensitivity['coverage_0.5'] == sample_totals['candidates'] == 1342
    assert sum(reasons.values()) == sample_totals['words']-sample_totals['candidates']
    audio_valid = np.asarray([r['audio_valid_dimensions'] for r in results['samples']]).sum(0)
    summary = {'samples':100, 'totals':dict(sample_totals),
               'coverage_groups':results['coverage_groups'],
               'macro_sample_coverage':float(np.mean([r['coverage'] for r in results['samples']])),
               'micro_word_coverage':sample_totals['candidates']/sample_totals['words'],
               'unresolved_direct_reasons':dict(reasons),
               'valid_audio_rows_by_dimension':audio_valid.tolist(),
               'sensitivity_candidates':sensitivity,
               'transformations':results['transformation_summary'],
               'acoustic_calibration':results['acoustic_calibration'],
               'results_sha256':digest(out/'results.json'),
               'scope':'Full attachment measurements and controlled robustness; no semantic accuracy or downstream sentiment gain.'}
    write(out/'summary.json', summary)
    lines = ['# 全量质量核查与受控稳定性实验', '',
        '本报告使用全部100条附件，生成与比较不读取人工记录或情感标签。受控实验保持内容不变，检验算法对音量和时间平移的响应；参考是原自动输出，不是人工词界真值。', '',
        '## 特征与关系可用性', '',
        f"1932词中1342词有候选，590词未确定。样本级：{results['coverage_groups']}；样本等权平均覆盖{summary['macro_sample_coverage']:.2%}，词等权覆盖{summary['micro_word_coverage']:.2%}。两者均不是准确率。", '',
        '| 未确定词直接原因（互斥） | 词数 |', '|---|---:|']
    for name, count in reasons.items():
        lines.append(f'| {name} | {count} |')
    lines += ['', '这些是算法证据缺失原因，不是对真实音轨内容的自动判定。尤其不能把“无唯一精确识别对应”全部解释为给定文本错误。', '',
        f"视觉采样帧共{sample_totals['video_rows']}：无人脸{sample_totals['no_face_frames']}，单人脸{sample_totals['single_face_frames']}，多人脸{sample_totals['multiple_face_frames']}。共{sample_totals['face_events']}个人脸事件，其中{sample_totals['crop_edge_events']}个带裁剪边缘标记。检出与未检出均不等于表情识别正确/错误。", '',
        '## 固定规则敏感性', '', '| 设置 | 候选词数 |', '|---|---:|']
    lines.extend(f'| {name} | {count} |' for name, count in sensitivity.items())
    lines += ['', '只改变列出的门槛，其余识别结果和规则相同。该表展示覆盖对规则的敏感性，不按候选更多选择最优方法。', '',
        '## 保持内容的音频变换', '',
        '| 变换 | 候选 | 保留原候选 | 丢失 | 新增 | 共同候选端点残差中位数/95分位（秒） |',
        '|---|---:|---:|---:|---:|---|']
    for name, v in results['transformation_summary'].items():
        e=v['boundary_residual_s']
        lines.append(f"| {name} | {v['candidates']} | {v['retained']} | {v['lost']} | {v['added']} | {e['median']:.6f}/{e['p95']:.6f} |")
    lines += ['', '前置静音实验在比较端点前减去已知1秒平移。端点统计只针对两次都有候选的词，因此必须同时报告保留与丢失，不能用小残差掩盖弃权。新增候选没有因此获得正确性认证。本实验不测WER、自然误对应率或绝对词界误差。', '',
        '## 声学量的解析信号校核', '',
        '| 已知频率（Hz） | 有效音高帧 | 绝对音高误差中位数/95分位（音分） |', '|---|---:|---|']
    for row in results['acoustic_calibration']['tones']:
        e=row['absolute_pitch_error_cents']
        lines.append(f"| {row['frequency_hz']} | {row['valid_pitch_frames']} | {e['median']:.4f}/{e['p95']:.4f} |")
    lines += ['', f"幅度减半时对数RMS相对理论变化ln(0.5)的最大绝对残差为{results['acoustic_calibration']['half_gain_log_rms_max_error']:.3g}。这校核量纲与实现，不证明自然语音音高准确率或情感预测能力。", '',
        '## 复现', '',
        '运行`evaluate_pipeline_quality.py --models 固定模型缓存 --output 新目录`得到冻结协议和逐条结果；将该目录作为`data/quality_evaluation`后运行`build_quality_report.py`。脚本、配置和结果摘要均随记录保存。', '']
    (out/'report.md').write_text('\n'.join(lines))
    print(summary['unresolved_direct_reasons'], sensitivity)


if __name__ == '__main__':
    main()
