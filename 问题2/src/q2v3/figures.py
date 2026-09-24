"""Paper-sized scientific figures; all numeric panels use saved evidence."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from .analysis import read,rows,arrays
from q2.data import digest,save_json

BLUE='#0072B2';ORANGE='#D55E00';GREEN='#009E73';GRAY='#666666'
def main(root):
    root=Path(root);out=root/'figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,
       'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,
       'axes.linewidth':.7,'lines.linewidth':1.6,'savefig.facecolor':'white','pdf.fonttype':42,'svg.fonttype':'path'})
    s=read(root/'analysis/summary.json');seed=rows(root/'analysis/seed_metrics.csv');manifest=[]
    is_new=s['frozen']['stable_improvement']
    def save(fig,name,caption,sources):
        for ext in ['svg','pdf','png']:fig.savefig(out/f'{name}.{ext}',dpi=600,bbox_inches='tight',pad_inches=.10)
        # Small inspection preview is a QA intermediate, not a paper image.
        fig.savefig(out/f'{name}_preview.png',dpi=130,bbox_inches='tight',pad_inches=.10)
        plt.close(fig)
        manifest.append({'id':name,'caption':caption,'sources':{p:digest(root/p) for p in sources},
                        'formats':['svg','pdf','png'],'png_dpi':600})
    def values(role,scenario,metric):
        match='candidate' if role=='final' and is_new else 'reference'
        return np.array([float(r['value']) for r in seed if r['model']==match and r['scenario']==scenario and r['metric']==metric])
    def aggregate(role,scope,metric):
        if scope=='clean':return values(role,'clean',metric)
        from q2.missing import MAIN
        return np.stack([values(role,q,metric) for q in MAIN]).mean(0)
    # Figure 1: structure grounded in the frozen checkpoint config, no fictitious gate.
    fig,ax=plt.subplots(figsize=(6.8,3.3));ax.set(xlim=(0,10),ylim=(0,5));ax.axis('off')
    def box(x,y,w,h,text,color='#EDF4F8'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.03,rounding_size=.05',fc=color,ec='#596A76',lw=.8))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8.5)
    def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=9,color='#596A76',lw=.9))
    layers=12 if '12' in s['frozen']['model_name'] else 2
    box(.1,3.7,1.8,.75,'Token IDs\n50 positions');box(.1,2.1,1.8,.75,'Audio features\n50 × 74');box(.1,.5,1.8,.75,'Visual features\n50 × 35')
    box(2.3,.45,1.5,4.05,'Valid mask\n+\nObservation\nmask\n+\nSpan removal','#F5F5F5')
    box(4.2,3.5,2.3,1.2,f'{layers}-layer BERT\nMasked pooling\n768 → 128')
    box(4.2,2.,2.3,.9,'Masked mean\nProjection → 64');box(4.2,.4,2.3,.9,'Masked mean\nProjection → 64')
    box(6.9,1.3,1.4,2.2,'Fusion\n259 → 128\nAvailability\nincluded')
    box(8.7,3.0,1.1,1.1,'Softmax\n3 classes','#FCF0E9');box(8.7,1.0,1.1,1.1,'3 tanh\nIntensity','#FCF0E9')
    for y in [4.075,2.475,.875]:arrow((1.9,y),(2.3,y));arrow((3.8,y),(4.2,y));arrow((6.5,y),(6.9,2.4))
    arrow((8.3,2.65),(8.7,3.55));arrow((8.3,2.1),(8.7,1.55))
    ax.text(5,4.92,'Missing token IDs are removed BEFORE contextual encoding',ha='center',fontsize=9,color=BLUE)
    save(fig,'fig01_architecture','模型结构与编码前缺失处理。长度50为特征位置上限，不代表50秒。融合为拼接与可用比例输入，不包含动态门控。',['study/frozen.json','src/q2v2/model.py'])
    # Figure 2: seed variability and separate conditional bootstrap uncertainty.
    fig,axes=plt.subplots(2,2,figsize=(6.8,5.2),layout='constrained')
    for col,metric in enumerate(['macro_f1','mae']):
        ax=axes[0,col]
        for role,color,offset,label in [('reference',BLUE,-.10,'2-layer FP32'),('final',ORANGE,.10,'Selected FP32')]:
            for x,scope in enumerate(['clean','missing_average']):
                v=aggregate(role,scope,metric);ax.errorbar(x+offset,v.mean(),yerr=v.std(ddof=1),fmt='o',color=color,capsize=4,label=label if x==0 else None)
                ax.scatter(x+offset+np.linspace(-.035,.035,len(v)),v,s=15,fc='white',ec=color,zorder=4)
        ax.set_xticks([0,1],['Clean','9 missing scenes']);ax.set_ylabel('Macro-F1 ↑' if col==0 else 'MAE ↓');ax.set_title('(a) Seed mean ± SD' if col==0 else '(b) Seed mean ± SD')
        ax.grid(axis='y',alpha=.2);ax.legend(loc='best',frameon=False)
        ax=axes[1,col];intervals=[r for r in s['intervals'] if r['metric']==metric and r['scenario'] in ['clean','missing_average']]
        for y,r in enumerate(intervals):
            ax.plot([r['lower'],r['upper']],[y,y],color=ORANGE,lw=2)
            ax.scatter(r['difference'],y,color=ORANGE,s=25)
        ax.axvline(0,color=GRAY,lw=.8,ls='--');ax.set_yticks([0,1],['Clean','Missing avg.']);ax.set_ylim(-.5,1.5)
        ax.set_xlabel('Selected − reference');ax.set_title('(c) ΔF1 with 95% cluster CI' if col==0 else '(d) ΔMAE with 95% cluster CI');ax.grid(axis='x',alpha=.2)
    save(fig,'fig02_comparison','验证集模型比较。上排为空心逐种子点与均值±样本标准差，下排为固定种子42模型差异的95%视频分组Bootstrap区间。两种不确定性含义不同。',['analysis/seed_metrics.csv','analysis/paired_cluster_intervals.csv'])
    fig,axes=plt.subplots(2,3,figsize=(6.8,4.6),layout='constrained')
    for j,modality in enumerate(['text','audio','vision']):
        for i,metric in enumerate(['macro_f1','mae']):
            ax=axes[i,j]
            for role,color,marker,label in [('reference',BLUE,'s','2-layer FP32'),('final',ORANGE,'o','Selected FP32')]:
                vv=[values(role,'clean' if r==0 else f'{modality}_{r}_random',metric) for r in [0,10,30,50]]
                mean=np.array([v.mean() for v in vv]);sd=np.array([v.std(ddof=1) for v in vv])
                ax.plot([0,10,30,50],mean,marker=marker,color=color,label=label,ms=3.5,ls='--' if role=='reference' else '-')
                ax.fill_between([0,10,30,50],mean-sd,mean+sd,color=color,alpha=.12)
            ax.set_xticks([0,10,30,50]);ax.grid(alpha=.2);ax.set_title(modality.capitalize() if i==0 else '')
            if i==1:ax.set_xlabel('Missing positions (%)')
            if j==0:ax.set_ylabel('Macro-F1 ↑' if i==0 else 'MAE ↓')
    for i in range(2):
        limits=[a.get_ylim() for a in axes[i]];lo=min(q[0] for q in limits);hi=max(q[1] for q in limits)
        for a in axes[i]:a.set_ylim(lo,hi)
    axes[0,0].legend(frameon=False,fontsize=7)
    save(fig,'fig03_missing_rates','验证集随机连续缺失的比例响应。折线为三个种子的均值，阴影为±一个样本标准差；0%表示不额外施加缺失。各行坐标范围相同。',['analysis/seed_metrics.csv'])
    fig,axes=plt.subplots(2,3,figsize=(6.8,4.8),layout='constrained')
    mats={}
    for i,metric in enumerate(['macro_f1','mae']):
        for j,mod in enumerate(['text','audio','vision']):
            clean=s['valid']['final']['clean'][metric]
            mats[(i,j)]=np.array([[s['valid']['final'][f'{mod}_{r}_{shape}'][metric]-clean for r in [10,30,50]] for shape in ['front','middle','back','multi']])
        bound=max(abs(mats[(i,j)]).max() for j in range(3));bound=max(bound,.005)
        for j,mod in enumerate(['text','audio','vision']):
            ax=axes[i,j];a=mats[(i,j)];im=ax.imshow(a,cmap='RdBu_r',vmin=-bound,vmax=bound,aspect='auto')
            for y in range(4):
                for x in range(3):ax.text(x,y,f'{a[y,x]:+.3f}',ha='center',va='center',fontsize=8,color='white' if abs(a[y,x])>.65*bound else '#222222')
            ax.set_xticks([0,1,2],['10%','30%','50%']);ax.set_yticks(range(4),['Front','Middle','Back','Multi'] if j==0 else ['']*4)
            ax.set_title(f'{mod.capitalize()} / '+('ΔF1' if i==0 else 'ΔMAE'))
        fig.colorbar(im,ax=axes[i].tolist(),fraction=.022,pad=.015)
    save(fig,'fig04_missing_shapes','最终模型在验证集36类补充场景中的变化，均相对于同模型完整输入。上排ΔF1越负表示损失越大，下排ΔMAE越正表示损失越大。同一指标共用色标。此图为种子42描述性结果。',['analysis/summary.json'])
    fig,axes=plt.subplots(1,2,figsize=(6.8,3.3),layout='constrained')
    names=['bert2_text','bert2_mm','bert2_span','bert2_distill','bert2_random_ablation','bert4_span']
    labels=['2L text','2L multimodal','2L + span','2L + distill','2L random init','4L + span']
    old=root/'evidence/round2/runs'
    for y,name in enumerate(names):
        v=[read(old/f'{name}_fold{f}/metrics.json')['selection']['score'] for f in range(3)]
        axes[0].plot(v,[y]*3,'o',ms=3,color=BLUE,alpha=.65);axes[0].scatter(np.mean(v),y,marker='D',s=25,color=ORANGE)
    axes[0].set_yticks(range(len(names)),labels);axes[0].invert_yaxis();axes[0].set_xlabel('Internal selection score');axes[0].set_title('(a) Preserved 2L / 4L controls');axes[0].grid(axis='x',alpha=.2)
    entries=s['development']['entries'];labelmap={'teacher12_compact_reuse':'12L compact + span','bert12_full_span':'12L full + span','bert12_full_none':'12L full / no span'}
    for f in range(3):
        axes[1].plot([e['metrics'][f]['selection']['score'] for e in entries],range(len(entries)),color=GRAY,alpha=.4,lw=.8)
    for y,e in enumerate(entries):
        v=[m['selection']['score'] for m in e['metrics']];axes[1].scatter(v,[y]*3,color=BLUE,s=17);axes[1].scatter(np.mean(v),y,color=ORANGE,marker='D',s=25)
    axes[1].set_yticks(range(len(entries)),[labelmap[e['name']] for e in entries]);axes[1].invert_yaxis();axes[1].set_xlabel('Internal selection score');axes[1].set_title('(b) 12L paired folds');axes[1].grid(axis='x',alpha=.2)
    save(fig,'fig05_ablation','内部三折对照。蓝点为各折，橙色菱形为三折算术平均；右图灰线连接同一折。左图与右图训练预算不同，不能将其差异解释为单一因素的因果效应。',['analysis/summary.json','evidence/round2/study/development.json'])
    fig,axes=plt.subplots(1,2,figsize=(6.8,3.2),layout='constrained');m=s['valid']['final']['clean'];cm=np.array(m['confusion_matrix']);percent=cm/cm.sum(1,keepdims=True)
    axes[0].imshow(percent,vmin=0,vmax=1,cmap='Blues')
    for i in range(3):
        for j in range(3):axes[0].text(j,i,f'{cm[i,j]}\n{100*percent[i,j]:.1f}%',ha='center',va='center',color='white' if percent[i,j]>.55 else '#222222')
    labels=['Negative','Neutral','Positive'];axes[0].set_xticks(range(3),labels);axes[0].set_yticks(range(3),labels);axes[0].set_xlabel('Predicted class');axes[0].set_ylabel('True class');axes[0].set_title('(a) Count and row percentage')
    for role,color,offset,label in [('reference',BLUE,-.10,'2-layer FP32'),('final',ORANGE,.10,'Selected FP32')]:
        axes[1].scatter(s['valid'][role]['clean']['class_f1'],np.arange(3)+offset,color=color,label=label)
    axes[1].set_yticks(range(3),labels);axes[1].invert_yaxis();axes[1].set_xlim(0,1);axes[1].set_xlabel('Class F1');axes[1].set_title('(b) Class-specific F1');axes[1].legend(frameon=False);axes[1].grid(axis='x',alpha=.2)
    save(fig,'fig06_classification','最终模型种子42的完整验证集分类诊断。混淆矩阵同时标注计数和真实类别内百分比；类别F1与第二轮FP32基准使用相同样本。',['analysis/summary.json'])
    fig,axes=plt.subplots(1,2,figsize=(6.8,3.0),layout='constrained')
    source=[]
    for f,color in enumerate([BLUE,ORANGE,GREEN]):
        p=root/f'runs/bert12_full_span_fold{f}/epochs.json'
        if not p.exists():continue
        source.append(str(p.relative_to(root)));log=read(p)
        axes[0].plot([r['epoch'] for r in log],[r['loss'] for r in log],'-o',ms=3,color=color,label=f'Fold {f+1}')
        axes[1].plot([r['epoch'] for r in log],[r['score'] for r in log],'-o',ms=3,color=color,label=f'Fold {f+1}')
    for ax in axes:ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.legend(frameon=False)
    axes[0].set_ylabel('Training loss');axes[1].set_ylabel('Validation selection score');axes[0].set_title('(a) Training');axes[1].set_title('(b) Internal validation')
    save(fig,'fig07_learning','完整词表连续缺失模型的三折学习曲线。损失为训练样本平均监督损失，选择分数来自内部验证的固定九类缺失场景。',['study/protocol.json']+source)
    a=arrays(root/'evaluation/final/valid/predictions/clean.csv');res=a['predictions']-a['targets']
    fig,axes=plt.subplots(1,2,figsize=(6.8,3.0),layout='constrained')
    hb=axes[0].hexbin(a['targets'],a['predictions'],gridsize=22,mincnt=1,cmap='Blues',extent=(-3,3,-3,3))
    axes[0].plot([-3,3],[-3,3],'--',color=GRAY,lw=1);axes[0].set(xlabel='True intensity',ylabel='Predicted intensity',xlim=(-3,3),ylim=(-3,3));axes[0].set_aspect('equal');fig.colorbar(hb,ax=axes[0],label='Count',fraction=.05)
    axes[1].boxplot([res[a['labels']==c] for c in range(3)],tick_labels=labels,patch_artist=True,boxprops={'facecolor':'#DCEBF4'},medianprops={'color':ORANGE},flierprops={'marker':'.','markersize':3,'alpha':.5})
    axes[1].axhline(0,color=GRAY,ls='--',lw=.8);axes[1].set_ylabel('Prediction − truth');axes[1].grid(axis='y',alpha=.2)
    axes[0].set_title('(a) Intensity prediction');axes[1].set_title('(b) Residual by true class')
    save(fig,'fig08_regression','最终模型种子42的完整验证集回归诊断。左图为二维计数而非平滑拟合，虚线为理想预测；右图按真实类别展示残差，保留离群点。',['evaluation/final/valid/predictions/clean.csv'])
    save_json(out/'figure_manifest.json',manifest)
    (out/'图表说明.md').write_text('\n\n'.join(f"## {r['id']}\n\n{r['caption']}\n\n数据源："+'；'.join(r['sources']) for r in manifest),encoding='utf-8')
    print(f'Generated {len(manifest)} evidence-linked figure groups')
