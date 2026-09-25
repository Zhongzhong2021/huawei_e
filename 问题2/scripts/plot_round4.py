"""Data-bound round-four scientific figures; no manual experimental numbers."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from q2.data import digest, save_json
from q2v3.analysis import read, rows


def main(root):
    root = Path(root); out = root / 'figures'; out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10,
        'axes.labelsize': 9, 'legend.fontsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    palette = ['#0072B2', '#D55E00', '#009E73']; markers = ['o', 's', '^']; manifest = []
    def save(fig, name, caption, sources):
        for ext in ['svg', 'pdf', 'png']:
            fig.savefig(out / (name + '.' + ext), dpi=600, bbox_inches='tight')
        fig.savefig(out / (name + '_preview.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        manifest.append({'id': name, 'caption': caption, 'sources': {p: digest(root / p) for p in sources}, 'png_dpi': 600, 'formats': ['svg', 'pdf', 'png']})
    folds = rows(root / 'analysis/internal_folds.csv')
    names = ['reference', 'balanced_sqrt', 'balanced_inverse']
    labels = ['Unweighted', 'Power 0.5', 'Power 1.0']
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.9), layout='constrained')
    for axis, metric, title in zip(axs, ['score', 'neutral_f1'], ['(a) Missing-scene selection score', '(b) Neutral-class F1']):
        for fold in range(3):
            values = [float(next(r for r in folds if r['model'] == name and int(r['fold']) == fold)[metric]) for name in names]
            axis.plot(range(3), values, color='#999999', alpha=.65, lw=.9, zorder=1)
        for i, name in enumerate(names):
            values = [float(r[metric]) for r in folds if r['model'] == name]
            axis.scatter(np.full(len(values), i), values, color=palette[i], marker=markers[i], s=27, zorder=3)
            axis.scatter([i], [np.mean(values)], marker='_', s=280, color='black', zorder=4)
        axis.set(xticks=range(3), xticklabels=labels, title=title, ylabel='Score' if metric == 'score' else 'Neutral-class F1')
        axis.grid(axis='y', color='#eeeeee'); axis.set_axisbelow(True)
    fold_sizes = '、'.join(r['n'] for r in folds if r['model'] == 'reference')
    save(fig, 'r4_01_internal', f'图1 内部三折配对比较，折验证样本量依次为{fold_sizes}。每个彩色点代表一折，灰线连接同一折，黑色短线为三折算术均值；这些点不是独立重复实验或置信区间。分数与F1均越高越好。', ['analysis/internal_folds.csv', 'study/protocol.json'])
    conf = read(root / 'study/confirmed.json')
    if len(conf['records']) == 3:
        fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.8), layout='constrained')
        for ax, key, title in zip(axs, ['macro_f1', 'mae', 'neutral_f1'], ['(a) Clean Macro-F1', '(b) Clean MAE', '(c) Neutral F1']):
            get = lambda m: m['clean']['class_f1'][1] if key == 'neutral_f1' else m['clean'][key]
            values = np.array([[get(r['reference']), get(r['metrics'])] for r in conf['records']])
            for seed, pair in zip([42, 2026, 3407], values):
                ax.plot([0, 1], pair, color='#999999', lw=.8, alpha=.6)
            for i in range(2):
                ax.scatter(np.full(3, i), values[:, i], color=palette[i], marker=markers[i], s=25)
                ax.errorbar(i + .10, values[:, i].mean(), yerr=values[:, i].std(ddof=1), fmt='D', color=palette[i], ms=4, capsize=3)
            ax.set(xticks=[0, 1], xticklabels=['Round 3', 'Candidate'], title=title, xlim=(-.3, 1.35))
            ax.grid(axis='y', color='#eeeeee'); ax.set_axisbelow(True)
        save(fig, 'r4_02_seeds', '图2 官方验证集728条样本的三种子确认。圆点与方点为逐种子结果，连线保持相同种子配对，菱形与误差线为均值±样本标准差；标准差不是Bootstrap置信区间。F1越高越好，MAE越低越好。', ['study/confirmed.json'])
    if read(root / 'study/frozen.json')['stable_improvement']:
        fig, axs = plt.subplots(1, 2, figsize=(6.5, 3.0), layout='constrained')
        for ax, role, title in zip(axs, ['reference', 'final'], ['Round 3 reference', 'Frozen round 4']):
            cm = np.array(read(root / f'evaluation/{role}/valid/metrics.json')['clean']['confusion_matrix'])
            proportion = cm / cm.sum(1, keepdims=True)
            img = ax.imshow(proportion, vmin=0, vmax=1, cmap='Blues')
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f'{cm[i,j]}\n{proportion[i,j]:.1%}', ha='center', va='center', fontsize=9, color='white' if proportion[i,j] > .55 else 'black')
            ax.set(xticks=range(3), yticks=range(3), xticklabels=['Neg.', 'Neutral', 'Pos.'], yticklabels=['Neg.', 'Neutral', 'Pos.'], xlabel='Predicted class', ylabel='True class', title=title)
        fig.colorbar(img, ax=axs, shrink=.8, label='Within true-class proportion')
        save(fig, 'r4_03_confusion', '图3 固定种子42的完整官方验证集混淆矩阵，两个模型均使用相同728条样本。格内同时给出计数与真实类别内比例，两图使用共同色标。', ['evaluation/reference/valid/metrics.json', 'evaluation/final/valid/metrics.json'])
        seedrows = rows(root / 'analysis/seed_metrics.csv')
        fig, axs = plt.subplots(2, 3, figsize=(6.5, 4.25), layout='constrained', sharex=True, sharey='row')
        for j, (modality, title) in enumerate(zip(['text', 'audio', 'vision'], ['Text missing', 'Audio missing', 'Vision missing'])):
            for i, metric in enumerate(['macro_f1', 'mae']):
                ax = axs[i, j]
                for role, label, color, marker, linestyle in [('reference', 'Round 3', palette[0], 'o', '-'), ('candidate', 'Round 4', palette[1], 's', '--')]:
                    values = np.array([[float(next(r for r in seedrows if r['model'] == role and int(r['seed']) == seed and r['scenario'] == ('clean' if rate == 0 else f'{modality}_{rate}_random') and r['metric'] == metric)['value']) for rate in [0, 10, 30, 50]] for seed in [42, 2026, 3407]])
                    mean, sd = values.mean(0), values.std(0, ddof=1)
                    ax.plot([0, 10, 30, 50], mean, marker=marker, ls=linestyle, ms=3, color=color, label=label)
                    ax.fill_between([0, 10, 30, 50], mean - sd, mean + sd, color=color, alpha=.13)
                ax.grid(color='#eeeeee'); ax.set_axisbelow(True)
                if i == 0: ax.set_title(title)
                if j == 0: ax.set_ylabel('Macro-F1' if metric == 'macro_f1' else 'MAE')
                ax.set_xticks([0, 10, 30, 50])
        axs[0, 0].legend(frameon=False, loc='lower left')
        fig.supxlabel('Requested missing positions (%)', fontsize=9)
        save(fig, 'r4_04_missing', '图4 官方验证集连续缺失比例响应。线为三种子均值，阴影为±一个样本标准差，0%为完整输入；横轴是名义特征位置比例，非秒数。各行共用纵轴，保留非单调变化。Macro-F1越高越好，MAE越低越好。', ['analysis/seed_metrics.csv', 'analysis/actual_missing_rates.csv'])
    save_json(out / 'round4_figure_manifest.json', manifest)
    print(f'Generated {len(manifest)} verified-source figure groups', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path); main(p.parse_args().root)
