"""Reference-first candidate sequences, native features, and sample-clock review media."""
import argparse,json,time
from pathlib import Path
from fractions import Fraction
import numpy as np
from scipy.io import wavfile
import torch,torchaudio,mediapipe as mp
from transformers import AutoTokenizer,AutoModel,Wav2Vec2Processor,Wav2Vec2ForCTC
from torchvision.models import resnet18,ResNet18_Weights
from PIL import Image
from common import read,write,digest,run
from math_features import words,cells,acoustic,iou
from local_alignment import reference_in_hypothesis
from ctc_alignment import align_tokens

class Engine:
    def __init__(self,root,cfg):
        self.cfg=cfg;self.tokenizer=AutoTokenizer.from_pretrained(root/'text',local_files_only=True)
        self.text_model=AutoModel.from_pretrained(root/'text',local_files_only=True).eval()
        self.processor=Wav2Vec2Processor.from_pretrained(root/'ctc',local_files_only=True)
        self.ctc=Wav2Vec2ForCTC.from_pretrained(root/'ctc',local_files_only=True).eval()
        self.scene=resnet18(weights=None);self.scene.load_state_dict(torch.load(root/'resnet18.pth',weights_only=True,map_location='cpu'));self.scene.fc=torch.nn.Identity();self.scene.eval()
        self.transform=ResNet18_Weights.IMAGENET1K_V1.transforms()
        opts=mp.tasks.vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(root/'face_landmarker.task')),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,num_faces=cfg['face_max'],output_face_blendshapes=True,
            min_face_detection_confidence=cfg['face_confidence'],min_face_presence_confidence=cfg['face_confidence'])
        self.face=mp.tasks.vision.FaceLandmarker.create_from_options(opts);self.face_names=None

    @torch.inference_mode()
    def text(self,text,ws):
        b=self.tokenizer(text,return_offsets_mapping=True,return_tensors='pt',truncation=False);offsets=b.pop('offset_mapping')[0].numpy()
        assert b['input_ids'].shape[1]<=self.text_model.config.max_position_embeddings
        h=self.text_model(**b).last_hidden_state[0].numpy();features=[]
        for w in ws:
            l,r=w['char_span'];idx=np.flatnonzero((offsets[:,1]>l)&(offsets[:,0]<r));assert len(idx)
            w['token_indices']=idx.tolist();features.append(h[idx].mean(0))
        return np.array(features,np.float32),offsets

    @torch.inference_mode()
    def align(self,wave,ws,start):
        n=len(ws);iv=np.zeros((n,2));scores=np.zeros(n,np.float32);mask=np.zeros(n,bool)
        empty={'status':'no_signal','ctc_text':'','local_comparison':None}
        if not np.any(wave):return iv,iv.copy(),scores,mask,empty
        batch=self.processor(wave,sampling_rate=self.cfg['audio_rate'],return_tensors='pt')
        lp=self.ctc(batch.input_values).logits.log_softmax(-1);asr=self.processor.batch_decode(lp.argmax(-1))[0]
        reference=[p for w in ws for p in w['parts']];hyp=[p for w in words(asr) for p in w['parts']]
        local=reference_in_hypothesis(reference,hyp);meta={'status':'candidate','ctc_text':asr,'local_comparison':local}
        match=dict(local['pairs']);cursor=0
        for w in ws:
            w['local_asr_exact']=all(j in match for j in range(cursor,cursor+len(w['parts'])));cursor+=len(w['parts'])
        if not all(w['supported'] for w in ws):meta['status']='unsupported_text';return iv,iv.copy(),scores,mask,meta
        vocab=self.processor.tokenizer.get_vocab();target='|'.join(reference)
        if any(c not in vocab for c in target):meta['status']='unsupported_text';return iv,iv.copy(),scores,mask,meta
        ids=[vocab[c] for c in target];minimum=len(ids)+sum(a==b for a,b in zip(ids,ids[1:]))
        if not ids or minimum>lp.shape[1]:meta['status']='path_infeasible';return iv,iv.copy(),scores,mask,meta
        stride=field=1
        for k,s in zip(self.ctc.config.conv_kernel,self.ctc.config.conv_stride):field+=(k-1)*stride;stride*=s
        assert (len(wave)-field)//stride+1==lp.shape[1]
        frame_cells=cells(start+(np.arange(lp.shape[1])*stride+(field-1)/2)/self.cfg['audio_rate'],start,start+len(wave)/self.cfg['audio_rate'])
        def force(star):
            spans=align_tokens(lp,ids,outer=star)
            out=np.zeros((n,2));confidence=np.zeros(n,np.float32);pos=0
            for i,w in enumerate(ws):
                length=len('|'.join(w['parts']));g=spans[pos:pos+length];out[i]=[frame_cells[g[0].start,0],frame_cells[g[-1].end-1,1]]
                chars=[s for s in g if s.token!=vocab['|']];confidence[i]=sum(float(s.score)*(s.end-s.start) for s in chars)/sum(s.end-s.start for s in chars);pos+=length+1
            return out,confidence
        plain,_=force(False);outer,scores=force(True);mask[:]=True
        meta.update(outer_wildcard_allows_empty=True,stride=stride,receptive_field=field,ctc_frames=lp.shape[1],candidate_target_span=[float(outer[0,0]),float(outer[-1,1])])
        return outer,plain,scores,mask,meta

    @torch.inference_mode()
    def visual(self,path,native):
        cfg=self.cfg;tb=Fraction(native['time_base']);pts=np.array([float(t*tb) for t in native['pts_ticks']])
        indices=np.unique(np.r_[np.searchsorted(pts,np.arange(pts[0],pts[-1]+1e-10,cfg['video_step_s'])),len(pts)-1]);indices=indices[indices<len(pts)]
        width=cfg['image_width'];height=2*round(native['height']/native['width']*width/2)
        expression='+'.join(f'eq(n\\,{i})' for i in indices)
        raw=run(['ffmpeg','-nostdin','-v','error','-threads','1','-noautorotate','-i',str(path),'-vf',f'select={expression},scale={width}:{height}','-vsync','0','-pix_fmt','rgb24','-f','rawvideo','-'])
        frames=np.frombuffer(raw,np.uint8).reshape(-1,height,width,3);assert len(frames)==len(indices)
        scene=[];face_rows=[];box_rows=[];face_frame=[];face_region=[];edge_flags=[];counts=[]
        for b in range(0,len(frames),16):scene.append(self.scene(torch.stack([self.transform(Image.fromarray(f)) for f in frames[b:b+16]])).numpy())
        for i,frame in enumerate(frames):
            def detect(rect,region):
                x0,y0,x1,y1=rect;crop=np.ascontiguousarray(frame[y0:y1,x0:x1]);r=self.face.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=crop));found=[]
                for pts3,blend in zip(r.face_landmarks,r.face_blendshapes):
                    xy=np.array([[p.x,p.y] for p in pts3]);local=np.r_[xy.min(0),xy.max(0)]
                    edge=bool(np.any(local[:2]<=0) or np.any(local[2:]>=1))
                    box=(local*np.array([x1-x0,y1-y0,x1-x0,y1-y0])+np.array([x0,y0,x0,y0]))/np.array([width,height,width,height]);box=np.clip(box,0,1)
                    ordered=sorted(blend,key=lambda x:x.index);names=[v.category_name for v in ordered]
                    if self.face_names is None:self.face_names=names
                    assert names==self.face_names and len(names)==52
                    found.append((box,np.array([v.score for v in ordered],np.float32),region,edge))
                return found
            found=detect((0,0,width,height),-1)
            if not found:
                for region,(l,t,r,b) in enumerate(cfg['crop_fallback']):found.extend(detect((int(l*width),int(t*height),int(r*width),int(b*height)),region))
            found.sort(key=lambda x:(x[3],-float(np.prod(x[0][2:]-x[0][:2]))));kept=[]
            for f in found:
                if all(iou(f[0],k[0])<cfg['face_duplicate_iou'] for k in kept):kept.append(f)
            counts.append(len(kept))
            for box,values,region,edge in kept:box_rows.append(box);face_rows.append(values);face_frame.append(i);face_region.append(region);edge_flags.append(edge)
        arrays={'scene':np.concatenate(scene).astype(np.float32),'video_frame_indices':indices.astype(np.int64),'video_times':pts[indices],
            'video_cells':cells(pts[indices],native['start_s'],native['end_s']),
            'faces':np.array(face_rows,np.float32).reshape(-1,52),'face_boxes':np.array(box_rows,np.float32).reshape(-1,4),
            'face_frame_rows':np.array(face_frame,np.int64),'face_crop_regions':np.array(face_region,np.int16),
            'face_touches_crop_edge':np.array(edge_flags,bool),'face_counts':np.array(counts,np.int16)}
        return arrays

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--models',type=Path,required=True);ap.add_argument('--media',type=Path,required=True)
    args=ap.parse_args();inputs=read('data/inputs.json');cfg=read('config/method.json');root=Path('data/features');root.mkdir(parents=True,exist_ok=False);args.media.mkdir(parents=True,exist_ok=False)
    manifest=read(args.models/'manifest.json')
    for m in manifest:assert digest(args.models/m['path'])==m['sha256']
    write('data/models.json',manifest);write('data/execution.json',{'scripts':{p.name:digest(p) for p in Path('scripts').glob('*.py')},'torch':torch.__version__,'started':time.time()})
    torch.set_num_threads(2);torch.manual_seed(0);engine=Engine(args.models,cfg);stats=[]
    for item in inputs['samples']:
        tic=time.monotonic();sid=item['sample_id'];path=Path(inputs['source_root'])/item['path'];assert digest(path)==item['sha256'];ws=words(item['text'])
        rate=int(item['audio']['sample_rate']);channels=item['audio']['channels'];start=item['audio']['start_s']
        pcm=np.frombuffer(run(['ffmpeg','-nostdin','-v','error','-threads','1','-i',str(path),'-map','0:a:0','-c:a','pcm_f32le','-f','f32le','-']),'<f4').reshape(-1,channels)
        assert len(pcm)==item['pcm_samples'];mono=pcm.mean(1).copy()
        assert not np.any(pcm) or np.any(mono),'Channel cancellation needs explicit handling'
        wave=torchaudio.functional.resample(torch.from_numpy(mono),rate,cfg['audio_rate']).numpy()
        # Review WAV retains original rate, channel count and float PCM samples; no model resampling.
        wav= args.media/(sid+'.wav');wavfile.write(wav,rate,pcm)
        step=max(1,round(rate*.01));amplitude=np.max(np.abs(pcm),axis=1);peaks=[float(amplitude[i:i+step].max()) for i in range(0,len(pcm),step)]
        iv,plain,score,valid,align=engine.align(wave,ws,start);text,offsets=engine.text(item['text'],ws);a,amask,centres=acoustic(wave,cfg)
        arrays={'text':text,'text_char_offsets':offsets,'word_intervals':iv,'word_intervals_plain':plain,'word_scores':score,'word_candidate_mask':valid,
                'audio':a,'audio_dim_mask':amask,'audio_cells':cells(centres+start,start,start+len(wave)/cfg['audio_rate']) if len(centres) else np.zeros((0,2))}
        arrays.update(engine.visual(path,item['video']));assert all(np.isfinite(v).all() for v in arrays.values());assert digest(path)==item['sha256']
        np.savez_compressed(root/(sid+'.npz'),**arrays)
        meta={'sample_id':sid,'source_sha256':item['sha256'],'original_text':item['text'],'words':ws,'alignment':align,
            'audio_start_s':start,'original_rate':rate,'original_samples':len(pcm),'resampled_samples':len(wave),
            'review_wav':str(wav.resolve()),'review_wav_sha256':digest(wav),'waveform_step_samples':step,'waveform_peaks':peaks,
            'status':'candidate_not_human_validated','face_note':'Per-frame detections, not tracked speakers; boxes are normalized to full source image, crop-edge flag preserved.'}
        write(root/(sid+'.json'),meta);row={'sample_id':sid,'words':len(ws),'candidate_words':int(valid.sum()),'audio_frames':len(a),'video_frames':len(arrays['scene']),
            'frames_with_faces':int((arrays['face_counts']>0).sum()),'seconds':time.monotonic()-tic}
        stats.append(row);print(json.dumps(row),flush=True)
    engine.face.close();write('data/face_fields.json',engine.face_names);write('data/extraction.json',stats)
if __name__=='__main__':main()
