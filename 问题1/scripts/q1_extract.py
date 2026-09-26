# 本程序及代码是在人工智能工具辅助下完成的。
# 工具：OpenAI Codex；模型：GPT-6 Astra (gpt-6-astra)；开发机构：OpenAI；模型发布日期：2026-09-03。
# AI 来源说明于 2026-09-24 补录，未改变算法；参见 AI辅助使用记录与队内审阅要求.md。
"""Question 1: traceable raw-media features and CTC word alignment (no label use)."""
import argparse, csv, hashlib, importlib.metadata, itertools, json, os, re
import subprocess, sys, time, unicodedata
from pathlib import Path
import cv2
import librosa
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
import torchaudio
from num2words import num2words
from transformers import AutoModel, AutoTokenizer

SR, HOP, WIN, FFT = 16000, 320, 400, 512
ROOT = Path('/data/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条')
OUT = Path('results/q1')
BERT_LOCK = Path('results/text_encoder/encoder.lock.json')
FACE_MODEL = Path('/cache/q1/face_landmarker.task')
ACOUSTIC_NAMES = ([f'mfcc_{i}' for i in range(13)] +
                  [f'delta_mfcc_{i}' for i in range(13)] +
                  [f'delta2_mfcc_{i}' for i in range(13)] +
                  ['log_rms','zero_crossing_rate','spectral_centroid_hz',
                   'spectral_bandwidth_hz','spectral_rolloff85_hz','spectral_flatness',
                   'f0_hz','voiced_probability'])
BLENDSHAPE_NAMES = '_neutral browDownLeft browDownRight browInnerUp browOuterUpLeft browOuterUpRight cheekPuff cheekSquintLeft cheekSquintRight eyeBlinkLeft eyeBlinkRight eyeLookDownLeft eyeLookDownRight eyeLookInLeft eyeLookInRight eyeLookOutLeft eyeLookOutRight eyeLookUpLeft eyeLookUpRight eyeSquintLeft eyeSquintRight eyeWideLeft eyeWideRight jawForward jawLeft jawOpen jawRight mouthClose mouthDimpleLeft mouthDimpleRight mouthFrownLeft mouthFrownRight mouthFunnel mouthLeft mouthLowerDownLeft mouthLowerDownRight mouthPressLeft mouthPressRight mouthPucker mouthRight mouthRollLower mouthRollUpper mouthShrugLower mouthShrugUpper mouthSmileLeft mouthSmileRight mouthStretchLeft mouthStretchRight mouthUpperUpLeft mouthUpperUpRight noseSneerLeft noseSneerRight'.split()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def dump(path, data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

def command(args):
    return subprocess.check_output([str(a) for a in args])

def edit_distance(a,b):
    d=list(range(len(b)+1))
    for i,x in enumerate(a):
        q=[i+1]
        for j,y in enumerate(b): q.append(min(q[-1]+1,d[j+1]+1,d[j]+(x!=y)))
        d=q
    return d[-1]

def normalize(text):
    """Keep source offsets; expand spoken numbers/initialisms, remove only bracket metadata."""
    clean=text.replace('’',"'").replace('‘',"'")
    exclusions=[{'start':m.start(),'end':m.end(),'text':m.group(),'reason':'bracketed speaker metadata'}
                for m in re.finditer(r'\[[^\]]*\]',clean)]
    words=[]
    for m in re.finditer(r"\d+(?:,\d{3})*(?:st|nd|rd|th)?|[A-Za-z]+(?:'[A-Za-z]+)*",clean):
        if any(e['start']<=m.start()<e['end'] for e in exclusions): continue
        raw=m.group(); spoken=raw.lower(); rule='lowercase / apostrophe normalization'
        if raw[0].isdigit():
            digits=re.sub(r'\D','',raw)
            if re.search(r'(st|nd|rd|th)$',raw): spoken=num2words(int(digits),to='ordinal')
            elif len(digits)==4 and 1900<=int(digits)<2100:
                spoken=num2words(int(digits),to='year')
            else: spoken=num2words(int(digits))
            rule='English number expansion'
        elif raw in ('US','BIO'):
            spoken=' '.join(raw.lower()); rule='initialism spelled as letters'
        # Preserve questionable source spellings rather than silently correcting them.
        for piece in re.findall(r"[a-z]+(?:'[a-z]+)*",spoken):
            words.append({'word':piece,'source_text':raw,'char_start':m.start(),
                          'char_end':m.end(),'normalization':rule})
    return words,exclusions

def probe(path):
    meta=json.loads(command(['ffprobe','-v','error','-show_streams','-show_format','-of','json',path]))
    frames=json.loads(command(['ffprobe','-v','error','-select_streams','v:0','-show_frames',
                              '-show_entries','frame=best_effort_timestamp_time,pkt_duration_time',
                              '-of','json',path]))['frames']
    pts=np.array([float(f['best_effort_timestamp_time']) for f in frames],dtype=np.float64)
    assert len(pts)>1 and np.all(np.diff(pts)>0), 'non-monotonic source frame PTS'
    v=next(s for s in meta['streams'] if s['codec_type']=='video')
    a=next(s for s in meta['streams'] if s['codec_type']=='audio')
    origin=float(meta['format'].get('start_time',0))
    audio_start=float(a.get('start_time',origin))-origin
    video_pts=pts-origin
    return meta,v,a,video_pts,audio_start,origin

def audio_features(wave):
    # STFT uses a 25 ms Hann window inside a 32 ms FFT; centers every 20 ms.
    S=np.abs(librosa.stft(wave,n_fft=FFT,hop_length=HOP,win_length=WIN,window='hann',center=True,pad_mode='constant'))
    mel=librosa.feature.melspectrogram(S=S**2,sr=SR,n_mels=64,fmin=20,fmax=8000)
    mfcc=librosa.feature.mfcc(S=librosa.power_to_db(mel,ref=1.0,top_db=80),n_mfcc=13)
    delta=librosa.feature.delta(mfcc,width=9,mode='nearest')
    delta2=librosa.feature.delta(mfcc,width=9,order=2,mode='nearest')
    rms=librosa.feature.rms(y=wave,frame_length=WIN,hop_length=HOP,center=True,pad_mode='constant')[0]
    zcr=librosa.feature.zero_crossing_rate(wave,frame_length=WIN,hop_length=HOP,center=True)[0]
    centroid=librosa.feature.spectral_centroid(S=S,sr=SR)[0]
    bw=librosa.feature.spectral_bandwidth(S=S,sr=SR)[0]
    roll=librosa.feature.spectral_rolloff(S=S,sr=SR,roll_percent=.85)[0]
    flat=librosa.feature.spectral_flatness(S=S)[0]
    f0,voiced,vprob=librosa.pyin(wave,fmin=50,fmax=500,sr=SR,frame_length=1024,hop_length=HOP,
                                center=True,pad_mode='constant',fill_na=np.nan)
    n=S.shape[1]
    assert all(len(x)==n for x in (rms,zcr,f0,voiced,vprob))
    x=np.vstack([mfcc,delta,delta2,np.log(np.maximum(rms,1e-8)),zcr,centroid,bw,roll,flat,
                 np.nan_to_num(f0,nan=0),vprob]).T.astype(np.float32)
    centers=np.arange(n)*HOP/SR
    cells=np.c_[np.maximum(0,centers-HOP/SR/2),np.minimum(len(wave)/SR,centers+HOP/SR/2)]
    valid=cells[:,1]>cells[:,0]
    return x[valid],centers[valid],cells[valid],voiced[valid]

def pool(intervals,cells,x,valid,std=False):
    weights=np.maximum(0,np.minimum(intervals[:,None,1],cells[None,:,1])-
                       np.maximum(intervals[:,None,0],cells[None,:,0]))*valid[None,:]
    den=weights.sum(1); z=weights/np.maximum(den[:,None],1e-12)
    mean=z@x
    result=np.concatenate([mean,np.sqrt(np.maximum(z@(x*x)-mean*mean,0))],1) if std else mean
    coverage=den/np.maximum(intervals[:,1]-intervals[:,0],1e-12)
    return result.astype(np.float32),den>0,coverage.astype(np.float32)

def text_features(words,text,exclusions,tokenizer,model,device):
    # Keep punctuation and number spelling in BERT context; spoken expansions share source token spans.
    chars=list(text)
    for e in exclusions: chars[e['start']:e['end']]=[' ']*(e['end']-e['start'])
    clean=''.join(chars).replace('^',' ')
    enc=tokenizer(clean,return_tensors='pt',return_offsets_mapping=True,truncation=False)
    offsets=enc.pop('offset_mapping')[0].numpy()
    assert enc.input_ids.shape[1]<=512, 'needs contextual chunking, never silently truncate'
    with torch.inference_mode(): hidden=model(**{k:v.to(device) for k,v in enc.items()}).last_hidden_state[0].cpu().numpy()
    out=[]; membership=np.zeros((len(words),len(offsets)),bool)
    for j,w in enumerate(words):
        inds=[k for k,(a,b) in enumerate(offsets) if b>a and a<w['char_end'] and b>w['char_start']]
        assert inds
        w['bert_positions']=inds
        w['bert_tokens']=tokenizer.convert_ids_to_tokens(enc.input_ids[0,inds].tolist())
        membership[j,inds]=True
        out.append(hidden[inds].mean(0))
    return np.array(out,dtype=np.float32),enc.input_ids[0].numpy().astype(np.int32),membership,offsets.astype(np.int32)

def vision_features(path,pts,duration):
    sample_idx=np.unique(np.searchsorted(pts,np.arange(max(0,pts[0]),min(duration,pts[-1]+1e-6),.1)))
    sample_idx=sample_idx[sample_idx<len(pts)]
    wanted=set(sample_idx.tolist()); cap=cv2.VideoCapture(str(path)); rows=[]; times=[]; frame_ids=[]
    bboxes=[]; counts=[]; names=BLENDSHAPE_NAMES; thumbs=[]
    options=mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(FACE_MODEL)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,num_faces=3,
        min_face_detection_confidence=.5,min_face_presence_confidence=.5,
        output_face_blendshapes=True,output_facial_transformation_matrixes=True)
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
        for i in range(len(pts)):
            ok,bgr=cap.read()
            assert ok, f'decoding failed at original frame {i}'
            if i not in wanted: continue
            h,w=bgr.shape[:2]; scaled=cv2.resize(bgr,(round(w*min(1,640/w)),round(h*min(1,640/w))))
            rgb=cv2.cvtColor(scaled,cv2.COLOR_BGR2RGB)
            result=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb))
            counts.append(len(result.face_landmarks)); times.append(pts[i]); frame_ids.append(i)
            box=[]
            for landmarks in result.face_landmarks:
                xy=np.array([[l.x,l.y] for l in landmarks]); box.append(np.r_[xy.min(0),xy.max(0)])
            if box:
                k=int(np.argmax([(b[2]-b[0])*(b[3]-b[1]) for b in box]))
                blends=sorted(result.face_blendshapes[k],key=lambda c:c.index)
                local_names=[c.category_name for c in blends]
                if names is None:names=local_names
                assert names==local_names and len(blends)==52
                # Rotation matrix (9) avoids Euler-angle wrap-around; relative translation omitted.
                R=np.asarray(result.facial_transformation_matrixes[k])[:3,:3]
                row=np.r_[[c.score for c in blends],R.ravel()]
                bboxes.append(box[k])
            else:
                row=np.zeros(61);bboxes.append(np.zeros(4))
            rows.append(row)
            if len(thumbs)<6 and (len(rows)-1)%max(1,len(sample_idx)//6)==0:
                thumbs.append((i,float(pts[i]),scaled.copy(),bboxes[-1],counts[-1]))
        assert not cap.read()[0], 'ffprobe and decoder frame counts differ'
    cap.release()
    times=np.array(times); cuts=(times[:-1]+times[1:])/2
    cells=np.c_[np.r_[max(0,pts[0]),cuts],np.r_[cuts,min(duration,pts[-1]+np.median(np.diff(pts)))]]
    # Voronoi cells describe nearest-observed-frame support, not actual exposure duration.
    return {'vision_native':np.array(rows,np.float32),'vision_time':times,'vision_cells':cells,
            'vision_frame_index':np.array(frame_ids,np.int32),'vision_face_count':np.array(counts,np.int16),
            'vision_bbox':np.array(bboxes,np.float32),'vision_valid':np.array(counts)>0},names,thumbs

def main():
    global OUT
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='results/q1');ap.add_argument('--limit',type=int)
    ap.add_argument('--indices',nargs='*',type=int);ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    OUT=Path(args.output);(OUT/'samples').mkdir(parents=True,exist_ok=True);(OUT/'preview').mkdir(exist_ok=True)
    os.environ.setdefault('TORCH_HOME','/cache/torch'); torch.set_num_threads(8)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    lock=json.loads(BERT_LOCK.read_text());snapshot=lock['local_snapshot']
    for name,h in lock['files_sha256'].items(): assert sha(Path(snapshot)/name)==h
    tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
    bert=AutoModel.from_pretrained(snapshot,local_files_only=True,use_safetensors=True).eval().to(device)
    bundle=torchaudio.pipelines.MMS_FA;am=bundle.get_model(with_star=False).eval().to(device)
    aligner=bundle.get_aligner();tok=bundle.get_tokenizer();alphabet=bundle.get_labels(star=None)
    versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','torch','torchaudio','transformers',
                    'librosa','mediapipe','opencv-contrib-python','num2words']}
    config={'schema_version':'q1-v2','sample_rate':SR,'audio_hop_samples':HOP,'audio_window_samples':WIN,
            'audio_fft':FFT,'pitch_window_samples':1024,'pitch_range_hz':[50,500],'video_sample_interval_s':.1,
            'video_max_width':640,'face_max_count':3,'face_detection_threshold':.5,'face_presence_threshold':.5,
            'face_selection':'largest normalized landmark bounding box per frame; no active speaker assertion',
            'ctc_low_score_threshold':.3,'ctc_sample_cer_review_threshold':.25,
            'ctc_boundary_handling':'optional leading/trailing star with synthetic sentinel rows; strict baseline retained',
            'word_audio_pooling':'overlap weighted mean + std, including zero F0 for unvoiced cells',
            'word_vision_pooling':'overlap weighted mean of valid nearest-frame Voronoi cells; missing is zero+mask',
            'padding':'ragged on disk; collate right-zero-padding with separate length and modality masks',
            'device':device,'versions':versions,'python':sys.version,'ffmpeg':command(['ffmpeg','-version']).decode().splitlines()[0],
            'bert_model_id':lock['model_id'],'bert_revision':lock['revision'],'bert_files_sha256':lock['files_sha256'],
            'models':{'mms_fa':{'sha256':sha('/cache/torch/hub/checkpoints/model.pt'),
                              'url':'https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/model.pt'},
                      'face_landmarker':{'sha256':sha(FACE_MODEL),
                              'url':'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'}},
            'label_table_sha256':sha(ROOT/'label-100.xlsx'),'script_sha256':sha(__file__),
            'acoustic_native_columns':ACOUSTIC_NAMES,
            'acoustic_aligned_columns':['mean_'+s for s in ACOUSTIC_NAMES]+['std_'+s for s in ACOUSTIC_NAMES]}
    dump(OUT/'run_config.json',config)
    # Only these columns enter this pipeline. No sentiment labels or split datasets loaded.
    df=pd.read_excel(ROOT/'label-100.xlsx',usecols=['video_id','clip_id','text'])
    if args.indices is not None: df=df.iloc[args.indices]
    elif args.limit:df=df.head(args.limit)
    for index,r in df.iterrows():
        sid=f'{r.video_id}$_${int(r.clip_id)}';stem=f'{r.video_id}__{int(r.clip_id)}'
        dest=OUT/'samples'/stem;started=time.time()
        if args.resume and dest.with_suffix('.json').exists():continue
        path=ROOT/r.video_id/f'{int(r.clip_id)}.mp4'
        try:
            meta,v,a,pts,a_start,origin=probe(path)
            duration=float(meta['format']['duration'])
            raw=command(['ffmpeg','-v','error','-i',path,'-map','0:a:0','-ac','1','-ar',SR,'-f','f32le','pipe:1'])
            wave=np.frombuffer(raw,dtype='<f4').copy()
            words,exclusions=normalize(r.text)
            with torch.inference_mode():
                emission,_=am(torch.from_numpy(wave)[None].to(device))
                strict_spans=aligner(emission[0],tok([w['word'] for w in words]))
                # Missing speech before/after the supplied sentence must not stretch boundary words.
                ext=torch.cat([emission[0],torch.zeros_like(emission[0,:,:1])],1)
                sentinel=torch.full_like(ext[:1],-1e9);sentinel[:,-1]=0
                padded=torch.cat([sentinel,ext,sentinel],0)
                flexible=aligner(padded,tok(['*']+[w['word'] for w in words]+['*']))
                spans=flexible[1:-1]
            ratio=len(wave)/SR/emission.shape[1]
            greedy=''.join(alphabet[k] for k,_ in itertools.groupby(emission[0].argmax(-1).tolist()) if k!=0)
            reference=''.join(w['word'] for w in words)
            cer=edit_distance(reference,greedy)/max(len(reference),1)
            strict_intervals=np.array([[sp[0].start*ratio+a_start,sp[-1].end*ratio+a_start] for sp in strict_spans])
            for w,sp in zip(words,spans):
                w.update(start_s=(sp[0].start-1)*ratio+a_start,end_s=(sp[-1].end-1)*ratio+a_start,
                         score=float(sum(s.score*len(s) for s in sp)/sum(len(s) for s in sp)),
                         emission_start=sp[0].start-1,emission_end=sp[-1].end-1)
                w['review_required']=w['score']<.3
            intervals=np.array([[w['start_s'],w['end_s']] for w in words])
            T,bert_ids,bert_membership,bert_offsets=text_features(words,r.text,exclusions,tokenizer,bert,device)
            A,at,acells,voiced=audio_features(wave);at+=a_start;acells+=a_start
            V,vnames,thumbs=vision_features(path,pts,duration)
            Ap,amask,acover=pool(intervals,acells,A,np.ones(len(A),bool),std=True)
            Vp,vmask,vcover=pool(intervals,V['vision_cells'],V['vision_native'],V['vision_valid'])
            scores=np.array([w['score'] for w in words],np.float32)
            features={'text':T,'audio':Ap,'vision':Vp,'word_intervals':intervals,'alignment_score':scores,
                      'text_mask':np.ones(len(T),bool),'audio_mask':amask,'vision_mask':vmask,
                      'audio_coverage':acover,'vision_coverage':vcover,'alignment_review':scores<.3,
                      'strict_word_intervals':strict_intervals,
                      'bert_token_ids':bert_ids,'bert_word_membership':bert_membership,'bert_char_offsets':bert_offsets,
                      'audio_native':A,'audio_time':at,'audio_cells':acells,'audio_voiced':voiced,**V}
            for k,x in features.items():assert np.isfinite(x).all(), f'non-finite {k}'
            assert np.all(intervals[:,1]>intervals[:,0]) and np.all(intervals[1:,0]>=intervals[:-1,1])
            np.savez_compressed(dest.with_suffix('.npz'),**features)
            notes=[]
            if exclusions:notes.append('bracketed metadata excluded from speech')
            if cer>.25:notes.append('high greedy acoustic character error proxy; inspect transcript')
            if np.mean(V['vision_valid'])<.8:notes.append('face coverage below 80%; retain missing mask')
            if np.any(V['vision_face_count']>1):notes.append('multiple visible faces; selected largest, speaker identity unverified')
            info={'sample_id':sid,'table_row_0based':int(index),'source_path':path.relative_to(Path('/data')).as_posix(),
                  'source_sha256':sha(path),'text_original':r.text,'text_spoken_normalized':' '.join(w['word'] for w in words),
                  'excluded_spans':exclusions,'words':words,'notes':notes,'container_origin_s':origin,
                  'duration_s':duration,'audio_start_s':a_start,'decoded_audio_duration_s':len(wave)/SR,
                  'original_audio_rate':int(a['sample_rate']),'original_audio_channels':int(a['channels']),
                  'video_width':int(v['width']),'video_height':int(v['height']),'video_rate':v['avg_frame_rate'],
                  'video_stream_duration_s':float(v.get('duration',pts[-1])),
                  'audio_stream_duration_s':float(a.get('duration',len(wave)/SR)),
                  'declared_video_frames':int(v.get('nb_frames',len(pts))),
                  'original_video_frames':len(pts),'video_first_pts_s':float(pts[0]),'video_last_pts_s':float(pts[-1]),
                  'ctc_emission_frames':emission.shape[1],'ctc_seconds_per_frame':ratio,
                  'greedy_acoustic_transcript_unsegmented':greedy,'greedy_cer_proxy':cer,
                  'boundary_context_before_s':float(intervals[0,0]-a_start),
                  'boundary_context_after_s':float(a_start+len(wave)/SR-intervals[-1,1]),
                  'strict_first_word_interval':strict_intervals[0].tolist(),
                  'boundary_adjustment_max_s':float(np.max(np.abs(intervals-strict_intervals))),
                  'alignment_score_mean':float(scores.mean()),'low_score_words':int((scores<.3).sum()),
                  'word_count':len(words),'bert_token_count':len(bert_ids),'truncated_words':0,
                  'audio_native_frames':len(A),'video_sampled_frames':len(V['vision_time']),
                  'face_valid_frames':int(V['vision_valid'].sum()),'face_coverage':float(V['vision_valid'].mean()),
                  'multiple_face_frames':int((V['vision_face_count']>1).sum()),
                  'vision_valid_words':int(vmask.sum()),'vision_zero_coverage_words':int((~vmask).sum()),
                  'shape':{k:list(x.shape) for k,x in features.items()},
                  'feature_file':dest.with_suffix('.npz').name,'feature_sha256':sha(dest.with_suffix('.npz')),
                  'vision_columns':(vnames or [])+[f'rotation_{i}{j}' for i in range(3) for j in range(3)],
                  'elapsed_seconds':round(time.time()-started,3)}
            dump(dest.with_suffix('.json'),info)
            # A compact per-sample contact sheet enables visual quality review.
            tiles=[]
            for idx,t,bgr,bbox,count in thumbs:
                tile=cv2.resize(bgr,(320,180));x0,y0,x1,y1=bbox
                if count:cv2.rectangle(tile,(int(x0*320),int(y0*180)),(int(x1*320),int(y1*180)),(40,230,40),1)
                cv2.putText(tile,f'frame {idx} | {t:.3f}s | faces {count}',(5,17),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,255),1)
                tiles.append(tile)
            if tiles:cv2.imwrite(str(OUT/'preview'/f'{stem}.jpg'),np.hstack(tiles))
            print(json.dumps({'sample_id':sid,'words':len(words),'cer_proxy':round(cer,3),
                              'low_score':int((scores<.3).sum()),'face_coverage':info['face_coverage'],
                              'seconds':info['elapsed_seconds']},ensure_ascii=False),flush=True)
        except Exception as e:
            with (OUT/'errors.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'id':sid,'error':str(e)})+'\n')
            raise
    subprocess.run([sys.executable,'-m','pip','freeze'],stdout=(OUT/'environment.freeze.txt').open('w'),check=True)

if __name__=='__main__':main()
