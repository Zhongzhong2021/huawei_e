"""Describe feature availability and cross-recognizer content support separately."""
import argparse,json,re
from difflib import SequenceMatcher
from pathlib import Path
from common import read,write,digest
ROOT=Path(__file__).resolve().parents[2]

def coverage(reference,hypothesis):
    tokenize=lambda s:re.findall(r"[a-z]+(?:'[a-z]+)?",s.lower())
    a,b=tokenize(reference),tokenize(hypothesis)
    return sum(m.size for m in SequenceMatcher(None,a,b,autojunk=False).get_matching_blocks())/max(len(a),1)

def main():
 p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);a=p.parse_args()
 media=read(a.audit/'media_and_token_comparison.json');ctc=read(a.audit/'independent_ctc.json')
 visual={r['number']:r for r in media['attachment4_versions']};other={r['number']:r for r in ctc['samples']}
 inputs=read(ROOT/'results/evidence_navigation/inputs.json');rows=[]
 for s in inputs['samples']:
  sid=s['sample_id'];w=read(ROOT/'results/evidence_navigation/asr'/(sid+'.json'));c=other[sid]
  assert s['sha256']==w['source_sha256']==c['video_sha256']
  wt=' '.join(x['text'] for x in w['segments']);ct=c['ctc_text'];scores={'whisper':coverage(s['raw_text'],wt),'ctc':coverage(s['raw_text'],ct)}
  # A diagnostic flag, not an alignment gate, a fitted threshold or a truth label.
  concern=all(v<.5 for v in scores.values())
  rows.append({'id':sid,'video_sha256':s['sha256'],'raw_text':s['raw_text'],'whisper_text':wt,'ctc_text':ct,'ordered_lexical_coverage':scores,'content_status':'low_cross_recognizer_support' if concern else 'no_low_support_flag','content_verified':False,'feature_availability':visual[sid]})
 write(ROOT/'results/evidence_navigation/media_consistency.json',{'schema':'q3-media-consistency-v1','diagnostic_rule':'Both ordered lexical coverages below 0.5; not semantic accuracy and not a learned decision','human_verified':False,'ctc_model_files_sha256':ctc['model_files_sha256'],'samples':rows,'source_sha256':{n:digest(a.audit/n) for n in ['media_and_token_comparison.json','independent_ctc.json']},'script_sha256':digest(__file__)})
 print('Low content support:',[r['id'] for r in rows if r['content_status']=='low_cross_recognizer_support'])
if __name__=='__main__':main()
