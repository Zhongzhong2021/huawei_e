"""Plot paired fold-mean differences and fixed-fit video-bootstrap intervals."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--no-augmentation',type=Path,required=True)
    p.add_argument('--mixed',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    reports=[json.loads(x.read_text())['development'] for x in (a.no_augmentation,a.mixed)]
    groups=['clean','single','low_joint','medium_joint','heavy_joint','low_medium']
    labels=['Complete input','Single modality','Joint 10% / 20%','Joint 30%','Joint 50%','Joint 10%–30%']
    colors=['#B44A3A','#236A93']
    fig,axes=plt.subplots(1,2,figsize=(11.3,4.4),sharey=True)
    for ax,metric,title in zip(axes,['macro_f1','mae'],['Macro-F1 difference (higher is better)','MAE difference (lower is better)']):
        for i,(report,label,color) in enumerate(zip(reports,['No augmentation − span','Mixed joint − span'],colors)):
            rows=[next(r for r in report['paired_intervals'] if r['group']==g and r['metric']==metric) for g in groups]
            mean=np.array([r['difference'] for r in rows]);lo=np.array([r['lower'] for r in rows]);hi=np.array([r['upper'] for r in rows])
            ax.errorbar(mean,np.arange(len(groups))+(i-.5)*.22,xerr=[mean-lo,hi-mean],fmt='o',ms=4,capsize=3,color=color,label=label)
        ax.axvline(0,color='#555555',lw=.8)
        ax.set_title(title,fontsize=11)
        ax.grid(axis='x',alpha=.18)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.set_yticks(np.arange(len(groups)),labels)
        ax.spines[['right','top']].set_visible(False)
    axes[0].invert_yaxis()
    handles,labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.92),ncol=2,fontsize=9,frameon=False)
    fig.suptitle('Controlled three-fold comparison: fixed model, class weights and four training epochs',fontsize=12)
    fig.text(.5,.01,'Pointwise 95% paired video-bootstrap intervals; conditional on fitted models and masks.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.84])
    a.output.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf'):
        fig.savefig(a.output/f'augmentation_comparison.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':
    main()
