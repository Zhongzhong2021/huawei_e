"""Recompute reported comparisons from the shipped reference records (NumPy only)."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
W=Path(__file__).resolve().parent;F=Path(sys.argv[1]) if len(sys.argv)>1 else W.parent/'features/samples'
read=lambda n:json.loads((W/n).read_text(encoding='utf-8'))
rows=read('word_records.json');summary=read('word_summary.json');public=read('public_words.json');anchors={r['sample_id']:r for r in read('waveform_anchors.json')}
arrays={};word_checks=0
for r in rows:
    sid=r['sample_id'];vid=r['source_video'];p=F/(sid.replace('$_$','__')+'.npz')
    if sid not in arrays:arrays[sid]=np.load(p)
    z=arrays[sid];i=r['word_index'];ref=next(x for x in public[vid] if x['index']==r['public_index'] and x['word']==r['word'])
    assert np.isclose(r['public_start_local_s'],ref['start']-anchors[sid]['source_offset_s'],atol=1e-10,rtol=0)
    assert np.isclose(r['MMS_FA_start_s'],z['word_start_quantiles'][i,1],atol=1e-10,rtol=0)
    assert np.isclose(r['MMS_FA_end_s'],z['word_end_quantiles'][i,1],atol=1e-10,rtol=0)
    assert r['supported']==bool(z['alignment_supported'][i]);word_checks+=1
for cohort,condition in [('all_eligible',lambda r:r['eligible']),('current_supported',lambda r:r['eligible'] and r['supported']),('current_unsupported',lambda r:r['eligible'] and not r['supported'])]:
    rr=[r for r in rows if condition(r)]
    for method in ['MMS_FA','WAV2VEC2','uniform','equal_fusion']:
        a=np.array([[r[method+'_start_s']-r['public_start_local_s'],r[method+'_end_s']-r['public_end_local_s']] for r in rr]);target=summary[cohort][method]
        assert len(rr)==target['words'];assert np.isclose(abs(a).mean(),target['boundary_mae_s'])
        assert np.isclose(np.mean(np.max(abs(a),1)<=.1),target['within100ms'])
audio=[];voices=[];pitch=[]
for p in (W/'reference_frames').glob('*.npz'):
    z=np.load(p);m=z['time_eligible'];audio.extend((z['local_voiced'][m]==z['public_voiced'][m]).tolist());both=m&z['local_voiced']&z['public_voiced']&(z['public_f0']>0)&(z['local_f0']>0);pitch.extend((12*np.log2(z['local_f0'][both]/z['public_f0'][both])).tolist())
s=read('acoustic_summary.json');assert len(audio)==s['frames'];assert np.isclose(np.mean(audio),s['voicing_agreement']);assert len(pitch)==s['joint_voiced_frames'];assert np.isclose(np.median(abs(np.array(pitch))),s['pitch_abs_semitone_median'])
vision=[]
for p in (W/'visual_reference_frames').glob('*.npz'):
    z=np.load(p);m=z['time_eligible'];vision.extend((z['local_detected'][m]==z['public_success'][m]).tolist())
s=read('visual_summary.json')['clock_shift_sensitivity']['0'];assert len(vision)==s['frames'];assert np.isclose(np.mean(vision),s['agreement'])
count=0
for p in (W/'pitch_crosscheck').glob('*.npz'):
    z=np.load(p);assert all(np.isfinite(z[k]).all() for k in z.files);assert np.all(z['within_one_semitone']<=z['joint_voiced']);count+=1
assert count==100
result=dict(word_records_checked=word_checks,word_summary_recomputed=True,audio_frames_checked=len(audio),visual_frames_checked=len(vision),pitch_diagnostic_samples=count,passed=True)
(W/'recalculation_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
