"""Review Q1 source correspondence, feature meaning and exception handling."""
import argparse,csv,json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from q1v6_features import load_sample,collate

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--features',default='features');ap.add_argument('--diagnostics',default='evidence/q1_next/audit_100.csv');ap.add_argument('--output',default='review');a=ap.parse_args()
    F=Path(a.features);O=Path(a.output);O.mkdir(parents=True,exist_ok=True)
    with Path(a.diagnostics).open(encoding='utf-8-sig') as f:diagnostics={r['sample_id']:r for r in csv.DictReader(f)}
    files=sorted((F/'samples').glob('*.json'),key=lambda p:json.loads(p.read_text(encoding='utf-8'))['table_row_0based'])
    rows=[];grids=[];words=[];spans=0;ids=[];mappings=[]
    for p in files:
        z,m=load_sample(F,p.stem,'native');g,_=load_sample(F,p.stem,'clock');w,_=load_sample(F,p.stem,'word');grids.append(g);words.append(w);ids.append(m['sample_id'])
        W=len(m['words']);assert len(z['text'])==W
        for j,word in enumerate(m['words']):
            lo,hi=word['char_start'],word['char_end'];assert 0<=lo<hi<=len(m['text_original'])
            source=m['text_original'][lo:hi]
            assert source.replace('’',"'").replace('‘',"'")==word['source_text']
            mappings.append(dict(sample_id=m['sample_id'],word_index=j,char_start=lo,char_end=hi,original_fragment=source,apostrophe_normalized_fragment=word['source_text'],spoken_word=word['word'],bert_positions=json.dumps(word['bert_positions']),normalization=word['normalization']))
            pos=np.asarray(word['bert_positions']);assert len(pos)>0 and pos.min()>=0 and pos.max()<len(z['bert_token_ids'])
            off=z['bert_char_offsets'][pos];assert np.all((off[:,1]>lo)&(off[:,0]<hi))
            assert z['text_mask'][j];spans+=1
        for name in ['audio_cells','vision_cells','ctc_cells']:
            cells=z[name];assert np.all(cells[:,1]>=cells[:,0]);assert np.all(np.diff(cells[:,0])>=-1e-8)
        for prefix,mask in [('word_audio','word_audio_dim_mask'),('word_expression','word_expression_mask'),('word_scene','word_scene_mask')]:
            assert np.isfinite(z[prefix]).all() and np.all(z[prefix][~z[mask]]==0)
            assert not z[mask][~z['alignment_supported']].any()
        assert not np.any(z['word_audio_dim_mask'][:,45]&~z['word_audio_dim_mask'][:,0])
        vf=z['word_audio'][:,-1];assert np.all((vf>=0)&(vf<=1+1e-6))
        assert g['cells'][0,0]==0 and abs(g['cells'][-1,1]-m['duration_s'])<1e-9
        d=diagnostics[m['sample_id']];n=int(z['alignment_supported'].sum())
        assert n==int(d['enhanced_words']) and W==int(d['words'])
        rows.append(dict(sample_id=m['sample_id'],duration_s=m['duration_s'],words=W,supported_words=n,text_valid=int(z['text_mask'].sum()),word_audio_valid=int(z['word_audio_dim_mask'][:,0].sum()),word_pitch_valid=int(z['word_audio_dim_mask'][:,45].sum()),word_expression_valid=int(z['word_expression_mask'].sum()),word_scene_valid=int(z['word_scene_mask'].sum()),grid_rows=len(g['cells']),grid_audio_valid=int(g['audio_mask'].sum()),grid_expression_valid=int(g['expression_mask'].sum()),grid_scene_valid=int(g['scene_mask'].sum()),diagnostic_status=d['status']))
    assert len(ids)==100 and len(set(ids))==100 and set(ids)==set(diagnostics)
    for b in [collate(words),collate(grids)]:
        for k,v in b.items():
            if v.ndim>=2 and v.shape[:2]==b['sequence_mask'].shape:assert np.all(v[~b['sequence_mask']]==0)
    no=[r for r in rows if not r['supported_words']]
    result=dict(samples=100,source_word_spans_checked=spans,unique_ids=True,source_character_and_token_correspondence=True,ordered_time_cells=True,masks_and_padding_consistent=True,
        words=sum(r['words'] for r in rows),word_audio_valid=sum(r['word_audio_valid'] for r in rows),word_pitch_valid=sum(r['word_pitch_valid'] for r in rows),word_expression_valid=sum(r['word_expression_valid'] for r in rows),word_scene_valid=sum(r['word_scene_valid'] for r in rows),
        clips_with_grid_audio=sum(r['grid_audio_valid']>0 for r in rows),clips_with_grid_expression=sum(r['grid_expression_valid']>0 for r in rows),clips_with_grid_scene=sum(r['grid_scene_valid']>0 for r in rows),
        clips_no_supported_words=len(no),no_word_support_diagnostic_counts=dict(Counter(r['diagnostic_status'] for r in no)),no_word_support_with_grid_audio=sum(r['grid_audio_valid']>0 for r in no),no_word_support_with_grid_expression=sum(r['grid_expression_valid']>0 for r in no),no_word_support_with_grid_scene=sum(r['grid_scene_valid']>0 for r in no),
        note='Diagnostics describe fixed model outputs, not ground-truth language or mismatch annotations. No feature values, source texts, masks or thresholds changed.',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for name,data in [('全100条使用状态.csv',rows),('全1948词原文映射.csv',mappings)]:
        with (O/name).open('w',encoding='utf-8-sig',newline='') as f:writer=csv.DictWriter(f,fieldnames=data[0]);writer.writeheader();writer.writerows(data)
    result['apostrophe_normalized_source_fragments']=sum(r['original_fragment']!=r['apostrophe_normalized_fragment'] for r in mappings)
    (O/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
