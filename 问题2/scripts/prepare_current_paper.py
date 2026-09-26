"""Recompute frozen round-four paper evidence without training or model selection."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from audit_round6 import prediction, recompute


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main(source, runs, output):
    if output.exists():
        raise FileExistsError('Keep prior paper evidence; choose a new output root')
    frozen = read(source/'study/frozen.json')
    assert frozen['model_name'] == 'balanced_inverse' and frozen['stable_improvement']
    source_hashes = {}
    def track(path):
        source_hashes[str(path.resolve())] = digest(path)
        return path
    track(source/'study/frozen.json')
    main_scenes = [f'{m}_{r}_random' for m in ['text', 'audio', 'vision'] for r in [10, 30, 50]]
    extra_scenes = [f'{m}_{r}_{shape}' for m in ['text', 'audio', 'vision'] for r in [10, 30, 50] for shape in ['front', 'middle', 'back', 'multi']]
    validation = {}
    values = {}
    for role in ['reference', 'final']:
        saved = read(track(source/f'evaluation/{role}/valid/metrics.json'))
        validation[role] = {}
        for scene in ['clean']+main_scenes+extra_scenes:
            value = prediction(track(source/f'evaluation/{role}/valid/predictions/{scene}.csv'))
            validation[role][scene] = recompute(value, saved[scene])
            assert value[0] == (values[(role, 'clean')][0] if scene != 'clean' else value[0])
            if role == 'final':
                for j in [0, 1, 2]:
                    np.testing.assert_array_equal(value[j], values[('reference', scene)][j])
            values[(role, scene)] = value
    special_path = track(source/'final/attachment3_predictions.csv')
    special = rows(special_path)
    assert len(special) == len({r['sample_id'] for r in special}) == 30
    probs = np.array([[float(r['prob_'+c]) for c in ['negative', 'neutral', 'positive']] for r in special])
    strength = np.array([float(r['intensity']) for r in special])
    assert np.isfinite(probs).all() and ((probs >= 0) & (probs <= 1)).all()
    assert np.isfinite(strength).all() and np.max(abs(strength)) <= 3
    np.testing.assert_allclose(probs.sum(1), 1, atol=1e-6, rtol=0)
    np.testing.assert_array_equal(probs.argmax(1), [int(r['class_id']) for r in special])
    # Apply the already fixed case-selection rule, including the ID tie-break.
    ids, labels, targets, p, estimate = values[('final', 'clean')]
    classes = p.argmax(1)
    errors = abs(targets-estimate)
    cases = []
    for c in range(3):
        for kind in ['wrong_largest_mae', 'correct_median_mae']:
            choices = np.flatnonzero((labels == c) & ((classes != c) if kind.startswith('wrong') else (classes == c)))
            choices = sorted(choices, key=lambda i: (errors[i], ids[i]))
            if not choices:
                continue
            i = choices[-1] if kind.startswith('wrong') else choices[(len(choices)-1)//2]
            cases.append({'sample_id': ids[i], 'selection_rule': kind, 'true_class': c,
                          'predicted_class': int(classes[i]), 'true_intensity': float(targets[i]),
                          'predicted_intensity': float(estimate[i]), 'absolute_error': float(errors[i])})
    recorded_cases = rows(track(source/'analysis/case_selection.csv'))
    assert [c['sample_id'] for c in cases] == [c['sample_id'] for c in recorded_cases]
    epochs = [read(track(runs/f'balanced_inverse_fold{fold}/epochs.json')) for fold in range(3)]
    rates = rows(track(source/'analysis/actual_missing_rates.csv'))
    confirmed = read(track(source/'study/confirmed.json'))
    assert confirmed['stable_improvement'] and len(confirmed['records']) == 3
    tables = {'main_missing': [], 'shape_comparison': [], 'attachment3': special, 'cases': cases}
    clean = validation['final']['clean']
    for scene in main_scenes:
        m = validation['final'][scene]
        modality, rate, _ = scene.split('_')
        tables['main_missing'].append({'modality': modality, 'rate': int(rate), 'macro_f1': m['macro_f1'],
            'delta_f1': m['macro_f1']-clean['macro_f1'], 'mae': m['mae'], 'delta_mae': m['mae']-clean['mae']})
        other = validation['final'][f'{modality}_{rate}_multi']
        tables['shape_comparison'].append({'modality': modality, 'rate': int(rate),
            'single_f1': m['macro_f1'], 'multi_f1': other['macro_f1'],
            'delta_f1': other['macro_f1']-m['macro_f1'], 'delta_mae': other['mae']-m['mae']})
    summary = {'frozen': frozen, 'valid': validation, 'tables': tables, 'actual_missing_rates': rates,
               'confirmation': confirmed, 'training_epochs': epochs,
               'scope': 'Publication-only recomputation of frozen round4; no training, tuning, or new testing'}
    save(output/'analysis/current_evidence.json', summary)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10,
        'axes.labelsize': 9, 'axes.spines.top': False, 'axes.spines.right': False,
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    figures = []
    def figure(fig, name, caption):
        folder = output/'figures'
        folder.mkdir(exist_ok=True)
        for ext in ['svg', 'pdf', 'png']:
            fig.savefig(folder/f'{name}.{ext}', dpi=600, bbox_inches='tight', pad_inches=.08)
        fig.savefig(folder/f'{name}_preview.png', dpi=150, bbox_inches='tight', pad_inches=.08)
        plt.close(fig)
        figures.append({'id': name, 'caption': caption, 'model_sha256': frozen['checkpoint_sha256'],
                        'source': 'analysis/current_evidence.json', 'source_sha256': digest(output/'analysis/current_evidence.json'),
                        'script_sha256': digest(Path(__file__)), 'formats': ['svg', 'pdf', 'png'], 'png_dpi': 600})
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 4.6), layout='constrained')
    for row, metric in enumerate(['macro_f1', 'mae']):
        mats = [np.array([[validation['final'][f'{mod}_{rate}_{shape}'][metric]-clean[metric] for rate in [10, 30, 50]]
                         for shape in ['front', 'middle', 'back', 'multi']]) for mod in ['text', 'audio', 'vision']]
        bound = max(.005, max(np.max(abs(m)) for m in mats))
        for col, (mod, matrix) in enumerate(zip(['Text', 'Audio', 'Vision'], mats)):
            ax = axes[row, col]
            im = ax.imshow(matrix, vmin=-bound, vmax=bound, cmap='RdBu_r', aspect='auto')
            for y in range(4):
                for x in range(3):
                    ax.text(x, y, f'{matrix[y,x]:+.3f}', ha='center', va='center', fontsize=8,
                            color='white' if abs(matrix[y,x]) > .65*bound else 'black')
            ax.set(xticks=range(3), xticklabels=['10%', '30%', '50%'], yticks=range(4),
                   yticklabels=['Front', 'Middle', 'Back', 'Multi'] if col == 0 else ['', '', '', ''],
                   title=mod+' / '+('ΔMacro-F1' if row == 0 else 'ΔMAE'))
        fig.colorbar(im, ax=axes[row].tolist(), fraction=.024, pad=.02)
    figure(fig, 'current_missing_shapes', '第四轮最终模型种子42在官方验证集728条样本的36类补充缺失场景。数值为相对同模型完整输入的变化；上排ΔMacro-F1越负越差，下排ΔMAE越正越差。同一指标共用色标，位置比例不是时间。')
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0), layout='constrained')
    count = axes[0].hist2d(targets, estimate, bins=25, range=[[-3, 3], [-3, 3]], cmap='Blues', cmin=1)
    fig.colorbar(count[3], ax=axes[0], label='Sample count')
    axes[0].plot([-3, 3], [-3, 3], '--', color='#666666', lw=1)
    axes[0].set(xlabel='True intensity', ylabel='Predicted intensity', title='(a) Observed prediction counts')
    axes[1].boxplot([(estimate-targets)[labels == c] for c in range(3)], tick_labels=['Negative', 'Neutral', 'Positive'], showfliers=True)
    axes[1].axhline(0, color='#666666', ls='--', lw=1)
    axes[1].set(ylabel='Predicted − true intensity', title='(b) Residuals by true class')
    figure(fig, 'current_regression', '第四轮最终模型种子42的完整官方验证集回归诊断，n=728。左图为实际二维计数，虚线为理想预测；右图为各真实类别的残差箱线图，保留离群点。箱体表示第25至75百分位，线为中位数，须线采用1.5倍四分位距规则，不是置信区间。')
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), layout='constrained')
    for fold, history in enumerate(epochs):
        for ax, key, label in zip(axes, ['loss', 'score'], ['Training loss', 'Internal missing-scene score']):
            ax.plot([e['epoch'] for e in history], [e[key] for e in history],
                    marker=['o', 's', '^'][fold], color=['#0072B2', '#D55E00', '#009E73'][fold], label=f'Fold {fold+1}')
            ax.set(xlabel='Epoch', ylabel=label)
            ax.grid(axis='y', color='#eeeeee')
    axes[1].legend(frameon=False)
    figure(fig, 'current_learning', '第四轮逆频率加权方案的内部三折学习曲线。左图为日志中的训练损失，右图为九类固定缺失场景选择分数；各点来自实际运行轮次。三条曲线代表不同视频分组折，不是三个独立随机种子。')
    save(output/'figures/current_figure_manifest.json', figures)
    save(output/'audit/evidence_checks.json', {'verified_utc': datetime.now(timezone.utc).isoformat(),
         'verified_valid_prediction_files': 92, 'cases_match_recorded_rules': True, 'attachment3_rows': 30,
         'reference_final_pairs_match': True, 'source_hashes': source_hashes,
         'figure_count': len(figures), 'test_read': False, 'model_changed': False,
         'model_sha256': frozen['checkpoint_sha256'], 'script_sha256': digest(Path(__file__))})
    print(json.dumps({'verified_prediction_files': 92, 'figures': len(figures), 'attachment3_rows': 30}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for arg in ['source', 'runs', 'output']:
        p.add_argument('--'+arg, type=Path, required=True)
    a = p.parse_args()
    main(a.source, a.runs, a.output)
