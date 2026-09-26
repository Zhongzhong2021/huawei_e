"""Source spans, CTC time geometry, acoustic descriptors and spatial geometry."""
import re,unicodedata
import numpy as np
from scipy.fft import dct
from num2words import num2words

def words(text):
    out=[]
    for m in re.finditer(r"\d+(?:[.,]\d+)*|[^\W\d_]+(?:['’][^\W\d_]+)*",text,re.UNICODE):
        value=num2words(m[0].replace(',',''),lang='en') if m[0][0].isdigit() else m[0]
        value=unicodedata.normalize('NFKD',value.replace('’',"'")).upper()
        value=''.join(c for c in value if not unicodedata.combining(c));parts=re.findall(r"[A-Z]+(?:'[A-Z]+)*",value)
        supported=bool(parts) and not any(c.isalpha() and not 'A'<=c<='Z' for c in value)
        out.append({'text':m[0],'char_span':[m.start(),m.end()],'parts':parts,'supported':supported})
    return out

def cells(centres,start,end):
    centres=np.asarray(centres,dtype=float)
    if not len(centres) or np.any(np.diff(centres)<=0) or centres[0]<start or centres[-1]>end:raise ValueError('Invalid centres')
    edges=np.r_[start,(centres[:-1]+centres[1:])/2,end];return np.column_stack([edges[:-1],edges[1:]])

def iou(a,b):
    lo=np.maximum(a[:2],b[:2]);hi=np.minimum(a[2:],b[2:]);intersection=float(np.maximum(0,hi-lo).prod())
    union=float(np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-intersection)
    return intersection/union if union>0 else 0.

def acoustic(wave,cfg):
    rate=cfg['audio_rate'];window=cfg['audio_window'];hop=cfg['audio_hop']
    starts=np.arange(0,len(wave)-window+1,hop)
    if not len(starts):return np.zeros((0,17),np.float32),np.zeros((0,17),bool),np.zeros(0)
    frames=np.stack([wave[s:s+window] for s in starts]).astype(float)
    power=np.abs(np.fft.rfft(frames*np.hanning(window),n=512))**2;freq=np.fft.rfftfreq(512,1/rate)
    mel=lambda f:2595*np.log10(1+f/700)
    p=700*(10**(np.linspace(mel(0),mel(rate/2),42)/2595)-1)
    bank=np.array([np.maximum(0,np.minimum((freq-a)/(b-a),(c-freq)/(c-b))) for a,b,c in zip(p,p[1:],p[2:])])
    mfcc=dct(np.log(np.maximum(power@bank.T,1e-12)),type=2,norm='ortho',axis=1)[:,:13]
    rms=np.log(np.maximum(np.sqrt((frames**2).mean(1)),1e-12));zcr=((frames[:,:-1]>=0)!=(frames[:,1:]>=0)).mean(1)
    pitch=np.zeros(len(starts));period=np.zeros(len(starts));pmask=np.zeros(len(starts),bool);vmask=pmask.copy()
    lo=int(np.ceil(rate/cfg['pitch_range_hz'][1]));hi=int(np.floor(rate/cfg['pitch_range_hz'][0]));length=cfg['pitch_window']
    for i,s in enumerate(starts):
        begin=s+(window-1)//2-length//2
        if begin<0 or begin+length>len(wave):continue
        x=wave[begin:begin+length].astype(float);x-=x.mean()
        if x@x<=1e-16:continue
        corr=np.correlate(x,x,'full')[length-1:];corr/=corr[0];lag=lo+np.argmax(corr[lo:hi+1]);period[i]=corr[lag];pmask[i]=True
        if period[i]>=cfg['pitch_periodicity_min']:pitch[i]=rate/lag;vmask[i]=True
    data=np.column_stack([mfcc,rms,zcr,pitch,period]).astype(np.float32);mask=np.ones(data.shape,bool);mask[:,-2]=vmask;mask[:,-1]=pmask;data[~mask]=0
    return data,mask,(starts+(window-1)/2)/rate
