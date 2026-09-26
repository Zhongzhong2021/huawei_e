# 本程序及代码是在人工智能工具辅助下完成的。
# 工具：OpenAI Codex；模型：GPT-6 Astra (gpt-6-astra)；开发机构：OpenAI；模型发布日期：2026-09-03。
# AI 来源说明于 2026-09-24 补录，未改变算法；参见 AI辅助使用记录与队内审阅要求.md。
"""Verify full-sample new interface, immutable old evidence, and padding."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from q1_extract import sha,dump
from q1f_load import load_sample,collate

def main():
    root=Path('.');out=Path('results/q1_refinement_v1');checks=0;inputs={};samples=[];rows=[]
    for p in sorted((out/'faces').glob('*.json')):
        m=json.loads(p.read_text());x=load_sample(root,p.stem);old=np.load(Path('results/q1/samples')/p.with_suffix('.npz').name)
        gate=np.load(Path('results/q1_next/enhanced')/p.with_suffix('.npz').name)
        w=m['words'];n=m['frames']
        assert x['vision_native'].shape==(n,52) and x['vision_soft'].shape==(w,52);checks+=1
        assert x['text'].shape==(w,768) and x['audio_soft'].shape==(w,94) and x['scene_soft'].shape==(w,512);checks+=1
        assert np.array_equal(x['vision_frame_index'],old['vision_frame_index']) and np.array_equal(x['vision_time'],old['vision_time']) and np.array_equal(x['vision_cells'],old['vision_cells']);checks+=1
        assert np.array_equal(x['alignment_supported'],gate['alignment_supported']);checks+=1
        assert np.array_equal(x['vision_native'][old['vision_valid']],old['vision_native'][old['vision_valid'],:52]);checks+=1
        assert np.all(x['vision_native'][~x['vision_valid']]==0) and np.all(x['vision_valid']>=old['vision_valid']);checks+=1
        for k in ('audio_soft','vision_soft','scene_soft'):
            assert np.isfinite(x[k]).all() and np.all(x[k][~x[k+'_mask']]==0);checks+=1
        assert np.all(x['vision_soft_mask']<=x['alignment_supported']);checks+=1
        assert np.all(x['vision_soft_mask']>=gate['vision_soft_mask']);checks+=1
        with np.load(p.with_suffix('.npz')) as f:
            new=f['recovery_route']==2
            assert np.all(f['yunet_score'][new]>=.9) and np.all(f['roi_landmark_iou'][new]>=.3);checks+=1
        for folder in ('q1/samples','q1_next/posterior','q1_next/enhanced','q1_next/asr'):
            for suffix in ('.npz','.json'):
                t=Path('results')/folder/(p.stem+suffix)
                if t.exists():inputs[str(t)]=sha(t)
        samples.append(x)
    assert len(samples)==100;checks+=1
    for i in range(0,100,7):
        batch=samples[i:i+7];z=collate(batch)
        for j,s in enumerate(batch):
            n=len(s['text']);assert z['sequence_mask'][j].sum()==n;checks+=1
            for k in ('text','audio_soft','vision_soft','scene_soft'):
                assert np.array_equal(z[k][j,:n],s[k]) and np.all(z[k][j,n:]==0);checks+=1
    # The new full table uses the explicit 52-dimensional expression branch.
    face=pd.read_csv(out/'face_ablation_100.csv').set_index('sample_id')
    base=pd.read_csv('results/q1_rigor/all_100_feature_summary.csv')
    newtext=pd.read_csv(out/'text_comparison_100.csv').set_index('sample_id')
    for r in base.to_dict('records'):
        a=face.loc[r['sample_id']];r['face_native_shape']=f"{a['frames']}x52";r['face_word_shape']=f"{a['words']}x52"
        r['face_supported_words']=int(a['new_word_visual']);r['visual_variant']='expression52_roi_v1 + scene512'
        r['number_equivalent_sentence_gate']=bool(newtext.loc[r['sample_id'],'number_equivalent_sentence_gate'])
        r['status_before_number_diagnostic']=r.pop('status')
        r['frame_expression_valid']=int(a['roi_hybrid_valid']);r['recovered_expression_frames']=int(a['recovered'])
        rows.append(r)
    pd.DataFrame(rows).to_csv(out/'all_100_feature_summary_v1.csv',index=False)
    dump(out/'validation.json',dict(samples=100,checks=checks,all_passed=True,baseline_time_masks_preserved=True,
        baseline_valid_expression_frames_bitwise_preserved=True,feature_dimensions=dict(text=768,audio=94,expression=52,scene=512)))
    dump(out/'input_manifest.json',inputs);print(json.dumps(dict(samples=100,checks=checks,all_passed=True)))

if __name__=='__main__':main()
