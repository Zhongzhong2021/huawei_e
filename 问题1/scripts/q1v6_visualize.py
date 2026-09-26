"""Scientific figures and source-frame evidence for Q1 v6."""
import json,csv,subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
P=Path('results/q1_deep_v6');O=P/'figures';O.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text())
def main():
    val=read(P/'validation_v2/results.json');clock=read(P/'media_clock/results.json');rows=clock['rows']
    with (P/'features_v2/全词音高与时间对照.csv').open(encoding='utf-8-sig') as f:pitch=[r for r in csv.DictReader(f) if r['pitch_valid']=='True']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'text.parse_math':False})
    fig,ax=plt.subplots(1,3,figsize=(14,4.4),layout='constrained')
    x=np.array([r['container_duration_s'] for r in rows]);y=np.array([r['observed_end_s'] for r in rows]);ax[0].scatter(x,y,s=22,alpha=.7);ax[0].plot([0,36],[0,36],'--',color='grey');ax[0].set(xlabel='Container-declared duration (s)',ylabel='Observed presentation endpoint (s)',title='A. Source duration audit / 100 clips')
    xx=np.array([float(r['old_zero_filled_F0_hz']) for r in pitch]);yy=np.array([float(r['new_conditional_F0_hz']) for r in pitch]);fr=np.array([float(r['voiced_fraction']) for r in pitch]);c=ax[1].scatter(xx,yy,c=fr,s=8,cmap='viridis',vmin=0,vmax=1);ax[1].plot([0,500],[0,500],'--',color='grey');ax[1].set(xlabel='Zero-filled weighted F0 (Hz)',ylabel='Voiced-conditional F0 (Hz)',title='B. Pitch semantics / 1,253 valid words');fig.colorbar(c,ax=ax[1],label='Voiced time fraction',shrink=.75)
    xx=np.arange(2)
    for j,model in enumerate(['MMS_FA','WAV2VEC2_ASR_BASE_960H']):
        rr=[next(r for r in val['perturbation_summaries'] if r['model']==model and r['condition']=='white_noise_20dB' and r['group']==g) for g in ['supported','unsupported']];rates=[100*r['over100ms']/r['n'] for r in rr]
        bars=ax[2].bar(xx+(j-.5)*.32,rates,.32,label='MMS' if j==0 else 'Wav2vec2')
        for b,r in zip(bars,rates):ax[2].text(b.get_x()+b.get_width()/2,r+.4,f'{r:.2f}%',ha='center',fontsize=8)
    ax[2].set(xticks=xx,xticklabels=['Supported (1464)','Withheld (484)'],ylabel='Words with boundary drift >100ms (%)',title='C. Actual waveform noise / 20dB SNR');ax[2].legend()
    for a in ax:a.grid(axis='y',alpha=.2)
    fig.savefig(O/'q1_results.png',dpi=180);plt.close(fig)
    # Predetermined example: first original table row, not chosen for favorable results.
    files=sorted((P/'features_v2/samples').glob('*.json'),key=lambda p:read(p)['table_row_0based']);p=files[0];m=read(p);z=np.load(p.with_suffix('.npz'));raw=Path('/data')/m['source_path']
    fm=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_entries','frame=best_effort_timestamp_time','-of','json',str(raw)]))['frames'];framepts=np.array([float(f['best_effort_timestamp_time']) for f in fm])
    wave=np.frombuffer(subprocess.check_output(['ffmpeg','-v','error','-i',str(raw),'-map','0:a:0','-ac','1','-ar','16000','-f','f32le','pipe:1']),'<f4')
    fig,ax=plt.subplots(4,1,figsize=(12,8),sharex=True,layout='constrained',height_ratios=[1,1.2,1,1])
    ax[0].plot(np.arange(0,len(wave),16)/16000,wave[::16],lw=.6,color='#315977');ax[0].set(ylabel='Amplitude',title=f"Source-linked example: {m['sample_id']} / effective {m['duration_s']:.3f}s")
    for j,w in enumerate(m['words']):
        a=z['word_start_quantiles'][j,1];b=z['word_end_quantiles'][j,1];level=j%3
        ax[1].plot([a,b],[level,level],lw=6,solid_capstyle='butt',color='#268a74' if z['alignment_supported'][j] else '#aaa');ax[1].text((a+b)/2,level+.16,w['word'],ha='center',fontsize=8)
    ax[1].set(ylim=(-.3,3),yticks=[],ylabel='CTC word spans')
    f0=z['audio_native'][:,45].copy();f0[~z['audio_voiced']]=np.nan;ax[2].plot(z['audio_time'],f0,color='#bc6a19',lw=1);ax[2].set(ylabel='Voiced F0 (Hz)')
    ax[3].plot(z['vision_time'],z['expression_native'][:,44],label='mouthSmileLeft');ax[3].plot(z['vision_time'],z['expression_native'][:,45],label='mouthSmileRight');ax[3].legend(loc='upper right',fontsize=8);ax[3].set(ylabel='Blendshape score',xlabel='Source media time (s)',xlim=(0,m['duration_s']))
    for a in ax:a.grid(alpha=.2)
    fig.savefig(O/'typical_timeline.png',dpi=180);plt.close(fig)
    choices=[0,len(m['words'])//2,len(m['words'])-1];fig,ax=plt.subplots(1,3,figsize=(12,3.4),layout='constrained');cap=cv2.VideoCapture(str(raw));example=[]
    for j,a in zip(choices,ax):
        center=(z['word_start_quantiles'][j,1]+z['word_end_quantiles'][j,1])/2;idx=int(np.argmin(abs(z['vision_time']-center)));fi=int(z['vision_frame_index'][idx]);cap.set(cv2.CAP_PROP_POS_FRAMES,fi);ok,bgr=cap.read();assert ok
        a.imshow(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));a.axis('off');box=z['vision_bbox'][idx]*[bgr.shape[1],bgr.shape[0],bgr.shape[1],bgr.shape[0]]
        if z['expression_valid'][idx]:a.add_patch(plt.Rectangle(box[:2],box[2]-box[0],box[3]-box[1],fill=False,ec='#19dfc2',lw=1.5))
        a.set_title(f"{m['words'][j]['word']} / frame {fi}\nPTS {z['vision_time'][idx]:.3f}s",fontsize=10)
    cap.release();fig.savefig(O/'typical_source_frames.png',dpi=160);plt.close(fig)
    for j,w in enumerate(m['words']):
        center=(z['word_start_quantiles'][j,1]+z['word_end_quantiles'][j,1])/2;idx=int(np.argmin(abs(z['vision_time']-center)))
        example.append(dict(word_index=j,word=w['word'],char_start=w['char_start'],char_end_exclusive=w['char_end'],start_s=float(z['word_start_quantiles'][j,1]),end_s=float(z['word_end_quantiles'][j,1]),reference_frame_index=int(z['vision_frame_index'][idx]),reference_frame_pts=float(z['vision_time'][idx]),text_dim=768,audio_dim=97,expression_dim=52,scene_dim=512,pitch_valid=bool(z['word_audio_dim_mask'][j,45]),voiced_F0_hz=float(z['word_audio'][j,45]),voiced_fraction=float(z['word_audio'][j,96]),expression_present=bool(z['word_expression_mask'][j])))
        lo=int(np.searchsorted(framepts,example[-1]['start_s'],side='left'));hi=int(np.searchsorted(framepts,example[-1]['end_s'],side='left'));example[-1].update(frame_start_index=lo,frame_end_index_exclusive=hi)
    (P/'typical_example.json').write_text(json.dumps(dict(sample_id=m['sample_id'],selection='First original table row; chosen before plotting, no label selection.',text=m['text_original'],source_path=m['source_path'],source_sha256=m['source_sha256'],words=example),ensure_ascii=False,indent=2),encoding='utf-8')
    print('FIGURES COMPLETE',m['sample_id'])
if __name__=='__main__':main()
