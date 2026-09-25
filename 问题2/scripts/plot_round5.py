"""Pair internal folds without treating folds as independent random-seed replications."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from q2.data import digest, save_json
from q2v3.analysis import rows, read


def main(root):
    root=Path(root);out=root/'figures';out.mkdir(exist_ok=True)
    values=rows(root/'analysis/internal_folds.csv');protocol=read(root/'study/protocol.json')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'legend.fontsize':8,
        'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
    names=['reference']+[c['name'] for c in protocol['candidates'] if any(r['model']==c['name'] for r in values)]
    styles={'reference':('Immediate','#0072B2','o'), 'defer1':('Delay 1','#D55E00','s'), 'defer2':('Delay 2','#009E73','^')}
    labels=[styles[n][0] for n in names];palette=[styles[n][1] for n in names];markers=[styles[n][2] for n in names];manifest=[]
    def get(name,key):return np.array([float(next(r for r in values if r['model']==name and int(r['fold'])==f)[key]) for f in range(3)])
    def save(fig,name,caption):
        for ext in ['svg','pdf','png']:fig.savefig(out/f'{name}.{ext}',dpi=600,bbox_inches='tight')
        fig.savefig(out/f'{name}_preview.png',dpi=150,bbox_inches='tight');plt.close(fig)
        manifest.append({'id':name,'caption':caption,'sources':{p:digest(root/p) for p in ['analysis/internal_folds.csv','study/protocol.json']},'png_dpi':600})
    fig,axes=plt.subplots(1,2,figsize=(6.5,2.9),layout='constrained')
    for ax,key,title in zip(axes,['score','positive_f1'],['(a) Missing-scene selection score','(b) Positive-class F1']):
        matrix=np.array([get(n,key) for n in names])
        for f in range(3):ax.plot(range(len(names)),matrix[:,f],color='#aaaaaa',lw=.9,zorder=1)
        for i,v in enumerate(matrix):
            ax.scatter(np.full(3,i),v,s=30,color=palette[i],marker=markers[i],zorder=3)
            ax.scatter(i,v.mean(),marker='_',s=280,color='black',zorder=4)
        ax.set(xticks=range(len(names)),xticklabels=labels[:len(names)],title=title,ylabel='Score' if key=='score' else 'Positive-class F1')
        ax.grid(axis='y',color='#eeeeee');ax.set_axisbelow(True)
    sizes='、'.join(r['n'] for r in values if r['model']=='reference')
    save(fig,'r5_01_paired',f'图1 内部三折配对结果，验证样本量依次为{sizes}。每个彩色点代表一折，灰线连接同折，黑色短线为三折均值。两指标越高越好；折间差异不是独立种子标准差或置信区间。')
    limits=protocol['guardrails']
    panels=[('score','Selection score',0),('clean_f1','Clean Macro-F1',-limits['clean_f1_max_drop']),('clean_mae','Clean MAE',limits['clean_mae_max_increase']),
        ('accuracy','Accuracy',-limits['accuracy_max_drop']),('neutral_f1','Neutral F1',-limits['neutral_f1_max_drop']),('positive_f1','Positive F1',0)]
    fig,axes=plt.subplots(2,3,figsize=(6.5,4.15),layout='constrained')
    for ax,(key,label,threshold) in zip(axes.flat,panels):
        ax.axhline(0,color='#888888',lw=.7)
        ax.axhline(threshold,color='#555555',ls='--',lw=.9,label='Acceptance boundary')
        for i,name in enumerate(names[1:]):
            diff=get(name,key)-get('reference',key)
            ax.scatter(np.full(3,i),diff,color=palette[i+1],marker=markers[i+1],s=24)
            ax.scatter(i,diff.mean(),color='black',marker='_',s=230,zorder=4)
        ax.set(xticks=range(len(names)-1),xticklabels=labels[1:len(names)],xlim=(-.35,len(names)-1-.65),title=label)
        ax.grid(axis='y',color='#eeeeee');ax.set_axisbelow(True)
    fig.supylabel('Candidate minus immediate weighting',fontsize=9)
    save(fig,'r5_02_tradeoffs','图2 相对立即加权对照的逐折指标差值。黑色短线为均值，虚线为预设均值保护边界；MAE越低越好，其余指标越高越好。S与正向F1要求严格改善，另外还须至少两折提高S。单项通过不代表候选通过全部条件。')
    save_json(out/'round5_figure_manifest.json',manifest)
    print(f'Generated {len(manifest)} figure groups',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
