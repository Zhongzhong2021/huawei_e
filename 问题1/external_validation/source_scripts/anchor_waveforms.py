import json,time,subprocess,hashlib
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import correlate,resample_poly
W=Path('/work');F=Path('/features/samples');D=Path('/data')
P=json.loads((W/'protocol.json').read_text());rule=P['anchor_acceptance']
def ncc(x,y):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64);n=len(y);yc=y-y.mean()
    num=correlate(x,yc,mode='valid',method='fft');s=np.r_[0,np.cumsum(x)];ss=np.r_[0,np.cumsum(x*x)]
    energy=np.maximum(0,ss[n:]-ss[:-n]-(s[n:]-s[:-n])**2/n)
    ey=np.sum(yc*yc);ok=energy>max(float(energy.max())*1e-8,n*1e-12)
    out=np.zeros_like(num)
    if ey>n*1e-12:out[ok]=num[ok]/np.sqrt(energy[ok]*ey)
    return np.clip(out,-1,1)
def peak(x,y,sr,origin=None):
    lo=0 if origin is None else max(0,int((origin-.12)*sr));hi=len(x) if origin is None else min(len(x),int((origin+.12)*sr)+len(y))
    c=ncc(x[lo:hi],y);j=int(np.argmax(np.abs(c)));return (j+lo)/sr,float(c[j]),c
cache={};rows=[];t0=time.perf_counter()
for p in sorted(F.glob('*.json'),key=lambda p:json.loads(p.read_text())['table_row_0based']):
    m=json.loads(p.read_text());vid=m['sample_id'].split('$_$')[0];path=W/'source_audio'/(vid+'.wav')
    if vid not in cache:
        x,sr=sf.read(path,dtype='float32');x=x.mean(1) if x.ndim==2 else x
        assert sr==16000;cache[vid]=(x,resample_poly(x,1,4))
    x,xc=cache[vid]
    raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(D/m['source_path']),'-vn','-ac','1','-ar','16000','-f','f32le','pipe:1'])
    y=np.frombuffer(raw,np.float32);yc=resample_poly(y,1,4);offset,coarse,c=peak(xc,yc,4000)
    j=round(offset*4000);other=np.abs(c).copy();other[max(0,j-2000):j+2001]=0;second=float(other.max())
    offset,score,_=peak(x,y,16000,offset)
    win=min(len(y)//3,32000);starts=[0,(len(y)-win)//2,len(y)-win];sub=[]
    for st in starts:
        t,s,_=peak(x,y[st:st+win],16000,offset+st/16000);sub.append(dict(local_start_s=st/16000,offset_s=t-st/16000,correlation=s))
    spread=max(z['offset_s'] for z in sub)-min(z['offset_s'] for z in sub)
    accepted=bool(abs(score)>=rule['correlation_at_least'] and abs(coarse)-second>=rule['gap_to_peak_outside_half_second_at_least'] and spread<=rule['three_window_offset_spread_at_most_s'])
    row=dict(sample_id=m['sample_id'],source_video=vid,source_offset_s=offset-m['audio_start_s'],waveform_offset_s=offset,correlation=score,coarse_correlation=coarse,alternative_peak=second,peak_gap=abs(coarse)-second,window_offset_spread_s=spread,window_checks=sub,accepted=accepted,audio_length_s=len(y)/16000,source_audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),local_video_sha256=m['source_sha256'])
    rows.append(row);print(m['sample_id'],round(offset,5),round(score,4),accepted,flush=True)
    (W/'waveform_anchors.json').write_text(json.dumps(rows,indent=2))
(W/'anchor_summary.json').write_text(json.dumps(dict(samples=len(rows),accepted=sum(r['accepted'] for r in rows),elapsed_s=time.perf_counter()-t0,correlation_min=min(abs(r['correlation']) for r in rows)),indent=2))
