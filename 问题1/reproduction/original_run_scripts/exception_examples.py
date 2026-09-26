import json,subprocess
from pathlib import Path
import numpy as np,cv2,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'/project/scripts');from q1v6_features import load_sample
P=Path('/project');O=Path('/work/examples');O.mkdir(exist_ok=True)
cases=[('-iRBcNs9oI8__7','Text / clip mismatch'),('-NFrJFQijFE__1','No detected face')];results=[]
plt.rcParams['text.parse_math']=False
for stem,title in cases:
 z,m=load_sample(P/'results/q1_deep_v6/features_v2',stem,'native');g,_=load_sample(P/'results/q1_deep_v6/features_v2',stem,'clock');a=json.loads((P/'results/q1_next/asr'/f'{stem}.json').read_text());src=Path('/data')/m['source_path']
 wave=np.frombuffer(subprocess.check_output(['ffmpeg','-v','error','-i',str(src),'-map','0:a:0','-ac','1','-ar','16000','-f','f32le','pipe:1']),'<f4')
 fig,ax=plt.subplots(2,1,figsize=(10,3.8),layout='constrained',sharex=True)
 ax[0].plot(np.arange(len(wave))[::16]/16000+m['audio_start_s'],wave[::16],lw=.7);ax[0].set(title=f'{title}: {m["sample_id"]}',ylabel='Amplitude')
 flags=np.vstack([g['text_mask'],g['audio_mask'],g['expression_mask'],g['scene_mask']]);ax[1].imshow(flags,aspect='auto',interpolation='nearest',extent=(0,m['duration_s'],3.5,-.5),cmap='Greens',vmin=0,vmax=1);ax[1].set(yticks=range(4),yticklabels=['Mapped text','Audio','Face expression','Scene'],xlabel='Clip time (s)');fig.savefig(O/(stem+'_availability.png'),dpi=160);plt.close(fig)
 indices=np.linspace(0,len(z['vision_frame_index'])-1,3).astype(int);cap=cv2.VideoCapture(str(src));fig,ax=plt.subplots(1,3,figsize=(10,3),layout='constrained');frames=[]
 for idx,axis in zip(indices,ax):
  fi=int(z['vision_frame_index'][idx]);cap.set(cv2.CAP_PROP_POS_FRAMES,fi);ok,bgr=cap.read();assert ok;axis.imshow(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));axis.axis('off');axis.set_title(f'frame {fi} / PTS {z["vision_time"][idx]:.3f}s',fontsize=9);frames.append(dict(frame_index=fi,pts=float(z['vision_time'][idx])))
 cap.release();fig.savefig(O/(stem+'_frames.png'),dpi=160);plt.close(fig)
 results.append(dict(sample_id=m['sample_id'],selection=title,official_text=m['text_original'],asr_text=a['transcript'],words=len(m['words']),word_support=int(z['alignment_supported'].sum()),grid_rows=len(g['cells']),audio_grid=int(g['audio_mask'].sum()),expression_grid=int(g['expression_mask'].sum()),scene_grid=int(g['scene_mask'].sum()),frames=frames))
(O/'examples.json').write_text(json.dumps(results,ensure_ascii=False,indent=2));print(json.dumps(results,ensure_ascii=False))
