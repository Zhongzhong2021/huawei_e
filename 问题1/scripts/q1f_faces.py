# 本程序及代码是在人工智能工具辅助下完成的。
# 工具：OpenAI Codex；模型：GPT-6 Astra (gpt-6-astra)；开发机构：OpenAI；模型发布日期：2026-09-03。
# AI 来源说明于 2026-09-24 补录，未改变算法；参见 AI辅助使用记录与队内审阅要求.md。
"""All-sample missing-face recovery with conservative ROI confirmation.
Keeps original features and timestamps; exports a versioned 52-D branch.
"""
import argparse, hashlib, json, time
from pathlib import Path
import cv2, mediapipe as mp, numpy as np, pandas as pd
from q1_extract import FACE_MODEL, BLENDSHAPE_NAMES, sha, command, dump
from q1n_posterior import soft_pool

ROOT=Path('results/q1_refinement_v1')

def options(threshold):
    return mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(FACE_MODEL)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,num_faces=3,
        min_face_detection_confidence=threshold,min_face_presence_confidence=threshold,
        output_face_blendshapes=True,output_facial_transformation_matrixes=False)

def detect(detector,bgr):
    rgb=np.ascontiguousarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
    return detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb))

def iou(a,b):
    lt=np.maximum(a[:2],b[:2]);rb=np.minimum(a[2:],b[2:]);inter=np.maximum(rb-lt,0).prod()
    return float(inter/max((a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-inter,1e-9))

def recover(bgr,yn,lm,cfg):
    h,w=bgr.shape[:2];scale=min(1.,cfg['yunet_max_width']/w)
    resized=cv2.resize(bgr,(round(w*scale),round(h*scale)))
    yn.setInputSize((resized.shape[1],resized.shape[0]));_,faces=yn.detect(resized)
    if faces is None:return None,0
    ordered=sorted(faces,key=lambda a:float(a[2]*a[3]),reverse=True)
    for f in ordered[:cfg['max_roi_candidates']]:
        x,y,bw,bh=f[:4]/scale
        if min(bw,bh)<cfg['roi_min_pixels']:continue
        side=int(np.ceil(max(bw,bh)*cfg['roi_square_expansion']))
        left=int(np.floor(x+bw/2-side/2));top=int(np.floor(y+bh/2-side/2))
        # A padded square preserves the same isotropic transform at image edges.
        patch=np.zeros((side,side,3),np.uint8)
        x0,y0=max(0,left),max(0,top);x1,y1=min(w,left+side),min(h,top+side)
        if x1<=x0 or y1<=y0:continue
        patch[y0-top:y1-top,x0-left:x1-left]=bgr[y0:y1,x0:x1]
        r=detect(lm,cv2.resize(patch,(cfg['roi_resize'],cfg['roi_resize'])))
        for k,landmarks in enumerate(r.face_landmarks):
            xy=np.array([[l.x*side+left,l.y*side+top] for l in landmarks])
            box=np.r_[xy.min(0),xy.max(0)]
            overlap=iou(box,np.array([x,y,x+bw,y+bh]))
            if overlap<cfg['landmark_box_detector_iou_min']:continue
            blend=sorted(r.face_blendshapes[k],key=lambda c:c.index)
            assert [c.category_name for c in blend]==BLENDSHAPE_NAMES
            feature=np.array([c.score for c in blend],np.float32)
            assert np.isfinite(feature).all() and feature.min()>=0 and feature.max()<=1
            return dict(feature=feature,box=box/np.array([w,h,w,h]),roi=np.array([left,top,side]),
                        score=float(f[-1]),iou=overlap,detector_box=np.array([x,y,x+bw,y+bh])/np.array([w,h,w,h])),len(faces)
    return None,len(faces)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int);ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    cv2.setNumThreads(2)
    protocol=Path('configs/q1_refinement_protocol.json');conf=json.loads(protocol.read_text());cfg=conf['face']
    ROOT.mkdir(exist_ok=True);(ROOT/'faces').mkdir(exist_ok=True);(ROOT/'preview').mkdir(exist_ok=True)
    lock=json.loads(Path('/cache/q1/yunet.lock.json').read_text());model=Path('/cache/q1')/cfg['yunet_model']
    assert sha(model)==lock['sha256']
    lock.update(protocol_sha256=sha(protocol),landmarker_sha256=sha(FACE_MODEL),script_sha256=sha(__file__),
                opencv=cv2.__version__,mediapipe=mp.__version__)
    lockpath=ROOT/'face_run.lock.json'
    if lockpath.exists():assert json.loads(lockpath.read_text())==lock,'Use a new version for changed methods'
    else:dump(lockpath,lock)
    yn=cv2.FaceDetectorYN.create(str(model),'',(320,320),cfg['yunet_score_threshold'],cfg['yunet_nms_threshold'],cfg['yunet_top_k'])
    files=sorted(Path('results/q1/samples').glob('*.json'),key=lambda p:json.loads(p.read_text())['table_row_0based'])[:args.limit]
    rows=[];pool_errors=[];decode_checks=[];processed_checks=0
    with mp.tasks.vision.FaceLandmarker.create_from_options(options(.5)) as lm, \
         mp.tasks.vision.FaceLandmarker.create_from_options(options(cfg['ablation_relaxed_fullframe_threshold'])) as relaxed:
        for p in files:
            destination=ROOT/'faces'/p.name
            if args.resume and destination.exists():rows.append(json.loads(destination.read_text()));continue
            started=time.time();m=json.loads(p.read_text());z=np.load(p.with_suffix('.npz'));n=len(z['vision_valid'])
            raw=Path('/data')/m['source_path'];assert sha(raw)==m['source_sha256']
            candidate=z['vision_native'][:,:52].copy();old=z['vision_valid'];valid=old.copy();route=np.where(old,1,0).astype(np.uint8)
            boxes=z['vision_bbox'].copy();rois=np.zeros((n,3),np.int32);scores=np.zeros(n,np.float32);ious=np.zeros(n,np.float32)
            relaxed_valid=old.copy();proposal_counts=np.zeros(n,np.int16)
            cap=cv2.VideoCapture(str(raw));wanted={int(v):i for i,v in enumerate(z['vision_frame_index'])}
            preview_ids=set([0,n//2,n-1]);thumbs=[];recovered_thumbs=[]
            for fi in range(m['original_video_frames']):
                ok,bgr=cap.read();assert ok
                if fi not in wanted:continue
                j=wanted[fi];h,w=bgr.shape[:2]
                if not old[j]:
                    scaled=cv2.resize(bgr,(round(w*min(1,640/w)),round(h*min(1,640/w))))
                    relaxed_valid[j]=bool(detect(relaxed,scaled).face_landmarks)
                    r,count=recover(bgr,yn,lm,cfg);proposal_counts[j]=count
                    if r:
                        candidate[j]=r['feature'];valid[j]=True;route[j]=2;boxes[j]=r['box'];rois[j]=r['roi'];scores[j]=r['score'];ious[j]=r['iou']
                if j in preview_ids or (route[j]==2 and len(recovered_thumbs)<3):
                    tile=cv2.resize(bgr,(480,270));a=boxes[j]*[480,270,480,270]
                    if valid[j]:cv2.rectangle(tile,tuple(a[:2].astype(int)),tuple(a[2:].astype(int)),(0,255,0) if old[j] else (0,180,255),2)
                    cv2.putText(tile,f"frame {fi} t={z['vision_time'][j]:.3f} old={int(old[j])} route={route[j]}",(5,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
                    if j in preview_ids:thumbs.append(tile)
                    elif route[j]==2:recovered_thumbs.append(tile)
                if p.stem in ('-hnBHBN8p5A__7','-iRBcNs9oI8__3','-mJ2ud6oKI8__2') and j in preview_ids:
                    ff=command(['ffmpeg','-v','error','-i',raw,'-vf',f'select=eq(n\\,{fi})','-frames:v','1','-f','rawvideo','-pix_fmt','bgr24','pipe:1'])
                    other=np.frombuffer(ff,np.uint8).reshape(bgr.shape);delta=np.abs(other.astype(float)-bgr)
                    decode_checks.append(dict(sample_id=m['sample_id'],frame=fi,mae=float(delta.mean()),max_difference=float(delta.max())))
                    assert delta.mean()<2,'Decoder disagreement requires investigation'
            assert not cap.read()[0];cap.release()
            assert np.array_equal(candidate[old],z['vision_native'][old,:52])
            assert np.all(candidate[~valid]==0) and np.all(valid>=old)
            post=np.load(Path('results/q1_next/posterior')/p.with_suffix('.npz').name)
            evidence=np.load(Path('results/q1_next/enhanced')/p.with_suffix('.npz').name)
            membership=post['word_time_membership'].astype(np.float64)
            pooled,present,den=soft_pool(membership,post['ctc_cells'],z['vision_cells'],candidate.astype(float),valid)
            mask=present&evidence['alignment_supported'];final=pooled.copy();final[~mask]=0
            for wi in sorted(set([0,len(pooled)//2,len(pooled)-1])):
                weight=np.zeros(n)
                for t,cell in enumerate(post['ctc_cells']):
                    weight+=membership[wi,t]*np.maximum(0,np.minimum(cell[1],z['vision_cells'][:,1])-np.maximum(cell[0],z['vision_cells'][:,0]))
                weight*=valid;expected=weight@candidate/max(weight.sum(),1e-12)
                if not present[wi]:expected[:]=0
                err=float(np.max(abs(expected-pooled[wi])));assert err<1e-6;pool_errors.append(err)
            processed_checks+=7
            np.savez_compressed(destination.with_suffix('.npz'),vision_expression_native=candidate,vision_expression_valid=valid,
                vision_expression_time=z['vision_time'],vision_expression_cells=z['vision_cells'],vision_expression_frame_index=z['vision_frame_index'],
                recovery_route=route,vision_expression_bbox=boxes,roi_left_top_side=rois,yunet_score=scores,roi_landmark_iou=ious,
                yunet_proposal_count=proposal_counts,relaxed_fullframe_valid=relaxed_valid,
                vision_expression_soft_candidate=pooled,vision_expression_soft_presence=present,vision_expression_effective_seconds=den.astype(np.float32),
                vision_expression_soft_mask=mask,vision_expression_soft=final)
            # Every clip preview uses deterministic original frames; recovered examples are separate slots.
            tiles=(thumbs+recovered_thumbs)[:6]
            while len(tiles)<6:tiles.append(np.zeros((270,480,3),np.uint8))
            cv2.imwrite(str(ROOT/'preview'/f'{p.stem}.jpg'),np.vstack([np.hstack(tiles[:3]),np.hstack(tiles[3:])]))
            row=dict(sample_id=m['sample_id'],frames=n,baseline_valid=int(old.sum()),relaxed_fullframe_valid=int(relaxed_valid.sum()),
                     roi_hybrid_valid=int(valid.sum()),recovered=int((route==2).sum()),
                     baseline_word_visual=int(evidence['vision_soft_mask'].sum()),new_word_visual=int(mask.sum()),
                     words=len(pooled),word_time_supported=int(evidence['alignment_supported'].sum()),
                     source_sha256=m['source_sha256'],baseline_npz_sha256=sha(p.with_suffix('.npz')),
                     posterior_npz_sha256=sha(Path('results/q1_next/posterior')/p.with_suffix('.npz').name),
                     gate_npz_sha256=sha(Path('results/q1_next/enhanced')/p.with_suffix('.npz').name),
                     output_sha256=sha(destination.with_suffix('.npz')),seconds=round(time.time()-started,3))
            dump(destination,row);rows.append(row);print(json.dumps({k:row[k] for k in ('sample_id','frames','baseline_valid','roi_hybrid_valid','recovered','seconds')}),flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'face_ablation_100.csv',index=False)
    summary=dict(samples=len(rows),frames=sum(r['frames'] for r in rows),
                 baseline_valid=sum(r['baseline_valid'] for r in rows),relaxed_fullframe_valid=sum(r['relaxed_fullframe_valid'] for r in rows),
                 roi_hybrid_valid=sum(r['roi_hybrid_valid'] for r in rows),recovered=sum(r['recovered'] for r in rows),
                 zero_face_clips_before=sum(r['baseline_valid']==0 for r in rows),zero_face_clips_after=sum(r['roi_hybrid_valid']==0 for r in rows),
                 baseline_word_visual=sum(r['baseline_word_visual'] for r in rows),new_word_visual=sum(r['new_word_visual'] for r in rows),
                 unchanged_word_time_supported=sum(r['word_time_supported'] for r in rows),
                 pool_recomputations=len(pool_errors),max_pool_error=max(pool_errors,default=0),
                 per_sample_checks=processed_checks,decoder_comparison=decode_checks,
                 interpretation='Coverage only; no independently annotated frame precision/recall or speaker identity.')
    dump(ROOT/'face_summary.json',summary);print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
