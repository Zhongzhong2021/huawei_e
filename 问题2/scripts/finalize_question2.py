"""Assemble one reproducible Q2 result version from the verified fixed-recipe fit."""
import argparse,csv,json,shutil,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from q2.data import digest,save_json
from q2.metrics import metrics
from q2v3.analysis import arrays
from q2v2.deploy import predict

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--run',type=Path,required=True);p.add_argument('--special',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 expected=json.loads((ROOT/'docs/retraining_seed42/model.json').read_text())
 model=a.run/'final/model.pt';scaler=a.run/'final/scaler.npz'
 assert digest(model)==expected['checkpoint_sha256'] and digest(scaler)==expected['scaler_sha256']
 a.output.mkdir(parents=True)
 checks=predict(model,scaler,a.special,a.output/'attachment3_predictions.csv','cuda')
 assert digest(a.output/'attachment3_predictions.csv')==digest(a.run/'final/attachment3_predictions.csv')
 scores=json.loads((a.run/'evaluation/valid/metrics.json').read_text())
 verified=[]
 for scenario,score in scores.items():
  if scenario=='selection':continue
  path=a.run/'evaluation/valid/predictions'/f'{scenario}.csv';v=arrays(path)
  actual=metrics(v['labels'],v['targets'],v['probabilities'],v['predictions'])
  for key in ('accuracy','macro_f1','mae','pearson'):assert abs(actual[key]-score[key])<1e-10
  verified.append({'scenario':scenario,'sha256':digest(path)})
 save_json(a.output/'validation_metrics.json',scores)
 data=arrays(a.run/'evaluation/valid/predictions/clean.csv')
 opposite=(data['classes']!=1)&(np.where(data['predictions']<0,0,np.where(data['predictions']>0,2,1))!=data['classes'])&(data['predictions']!=0)
 diagnostic={'neutral_count':int((data['labels']==1).sum()),'neutral_correct':int(((data['labels']==1)&(data['classes']==1)).sum()),'opposite_polarity_count':int(opposite.sum()),'neutral_nonzero_count':int(((data['classes']==1)&(data['predictions']!=0)).sum()),'class_mae':[float(np.mean(abs(data['targets'][data['labels']==c]-data['predictions'][data['labels']==c]))) for c in range(3)]}
 save_json(a.output/'diagnostics.json',diagnostic)
 colors=['#236A93','#BD6233','#428060'];names=['text','audio','vision']
 fig,axes=plt.subplots(2,3,figsize=(10,5.3),sharex=True,sharey='row')
 for j,mode in enumerate(names):
  for i,metric in enumerate(['macro_f1','mae']):
   axes[i,j].plot([0,10,30,50],[scores['clean'][metric]]+[scores[f'{mode}_{r}_random'][metric] for r in [10,30,50]],'o-',color=colors[j])
   axes[i,j].grid(alpha=.2);axes[i,j].set_xticks([0,10,30,50])
   if i==0:axes[i,j].set_title(mode.capitalize())
   else:axes[i,j].set_xlabel('Missing positions (%)')
   if j==0:axes[i,j].set_ylabel('Macro-F1' if i==0 else 'MAE')
 fig.tight_layout();fig.savefig(a.output/'missing.pdf');plt.close(fig)
 fig,axes=plt.subplots(2,3,figsize=(11,6))
 for j,mode in enumerate(names):
  for i,metric in enumerate(['macro_f1','mae']):
   values=np.array([[scores[f'{mode}_{r}_{shape}'][metric]-scores['clean'][metric] for r in [10,30,50]] for shape in ['front','middle','back','multi']])
   limit=max(abs(scores[f'{m}_{r}_{shape}'][metric]-scores['clean'][metric]) for m in names for r in [10,30,50] for shape in ['front','middle','back','multi'])
   im=axes[i,j].imshow(values,cmap='RdBu' if i==0 else 'RdBu_r',vmin=-limit,vmax=limit)
   axes[i,j].set_xticks(range(3),['10%','30%','50%']);axes[i,j].set_yticks(range(4),['Front','Middle','Back','Multiple'])
   axes[i,j].set_title(mode.capitalize()+(' ΔF1' if i==0 else ' ΔMAE'))
   for y in range(4):
    for x in range(3):axes[i,j].text(x,y,f'{values[y,x]:+.3f}',ha='center',va='center',fontsize=8,color='white' if abs(values[y,x])>limit*.65 else 'black')
   fig.colorbar(im,ax=axes[i,j],fraction=.04,pad=.04)
 fig.tight_layout();fig.savefig(a.output/'missing_shapes.pdf');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(10,4))
 cm=np.array(scores['clean']['confusion_matrix']);axes[0].imshow(cm,cmap='Blues')
 for y in range(3):
  for x in range(3):axes[0].text(x,y,f'{cm[y,x]}\n{cm[y,x]/cm[y].sum():.1%}',ha='center',va='center',color='white' if cm[y,x]>cm.max()*.6 else 'black')
 axes[0].set_xticks(range(3),['Negative','Neutral','Positive']);axes[0].set_yticks(range(3),['Negative','Neutral','Positive']);axes[0].set_xlabel('Predicted class');axes[0].set_ylabel('True class')
 im=axes[1].hexbin(data['targets'],data['predictions'],gridsize=25,mincnt=1,cmap='Blues');axes[1].plot([-3,3],[-3,3],'k--',lw=1);axes[1].set_xlabel('True intensity');axes[1].set_ylabel('Predicted intensity');fig.colorbar(im,ax=axes[1],label='Count')
 fig.tight_layout();fig.savefig(a.output/'errors.pdf');plt.close(fig)
 save_json(a.output/'model.json',{**expected,'version':'q2-final-span-seed42-v1','checkpoint':str(model),'scaler':str(scaler),'unknown_policy':'retain','validation_samples':728,'validation_conditions':46,'attachment3_samples':30,'source_run':str(a.run),'parameters_bytes':model.stat().st_size,'classification':'negative/neutral/positive','intensity_range':[-3,3]})
 save_json(a.output/'verification.json',{'validation_tables_recomputed':len(verified),'prediction_files':verified,'attachment3':checks,'finalization_script_sha256':digest(Path(__file__))})
 save_json(a.output/'files.json',{f.name:digest(f) for f in sorted(a.output.iterdir()) if f.name!='files.json'})
 print({'output':str(a.output),'clean':scores['clean'],'diagnostics':diagnostic},flush=True)
if __name__=='__main__':main()
