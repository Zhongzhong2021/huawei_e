"""Check attachment-only evidence and separately attributed human evaluation."""
from pathlib import Path
from common import read,digest
from delivery_reader import Dataset
ROOT=Path(__file__).resolve().parents[1]
def main():
    evidence=read(ROOT/'data/attachment_audit/checks.json');inputs={s['sample_id']:s for s in read(ROOT/'data/inputs.json')['samples']}
    assert evidence['input_inventory_sha256']==digest(ROOT/'data/inputs.json')
    assert evidence['generator_sha256']==digest(ROOT/'scripts/build_attachment_audit.py')
    assert evidence['external_reference_data_used'] is False and evidence['human_review_used'] is False
    assert len(evidence['rows'])==100 and {r['sample_id'] for r in evidence['rows']}==set(inputs)
    zero=[]
    for r in evidence['rows']:
        s=inputs[r['sample_id']];assert r['source_sha256']==s['sha256'] and r['pcm_frames']==s['pcm_samples']
        assert r['channels']==s['audio']['channels'] and r['sample_rate']==int(s['audio']['sample_rate'])
        assert r['all_zero']==s['zero_audio'];assert not r['decode_warnings']
        assert abs(r['pcm_duration_s']-s['pcm_samples']/r['sample_rate'])<1e-10
        if r['all_zero']:
            assert r['absolute_peak']==0 and r['ignore_editlist_check']['all_zero'];zero.append(r['sample_id'])
    assert len(zero)==2 and set(zero)==set(evidence['zero_audio_samples'])
    # Human content judgments remain evaluation evidence, never algorithmic diagnoses.
    review=read(ROOT/'data/independent_content_review.json')['samples']
    absent=[r['sample_id'] for r in review if r['content_relation']=='absent']
    absent += [r['sample_id'] for r in read(ROOT/'data/local_evidence_corroborated/evaluation.json')['previous_full_audio_mismatch']]
    assert len(absent)==len(set(absent))==6
    ds=Dataset(ROOT);totals=[]
    for ids in [zero,absent]:
        count=0
        for sid in ids:
            sample=ds.sample(sid)
            for i in range(len(sample.automatic['words'])):
                assert sample.word(i)['relation'] is None;count+=1
        totals.append(count)
    assert totals==[24,57]
    print('Attachment checks verified: 100 samples; 2 zero tracks/24 words; separately human-reviewed 6 absent samples/57 words have no candidates.')
if __name__=='__main__':main()
