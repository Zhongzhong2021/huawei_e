"""Check the PDF, edited-table provenance, key metrics and actual pagination.
Code edited with OpenAI Codex (GPT-6); model release date unavailable.
"""
import csv,hashlib,json,re,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];PAPER=ROOT/'Paper'
Q3=ROOT/('问题三' if (ROOT/'问题三').is_dir() else '问题3')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cells(line):return [x.strip() for x in line.strip().strip('|').split('|')]
def table(text,caption):
 block=text.split(caption,1)[1].strip().split('\n\n',1)[1].split('\n\n',1)[0]
 return [cells(x) for x in block.splitlines()[2:] if x.startswith('|')]
def check_results():
 m=[(PAPER/'manuscript'/f'question{i}.md').read_text() for i in range(1,4)]
 raw=(ROOT/'问题1/问题1正文稿.md').read_text().split('## 9 全100条结果表')[1]
 old={cells(l)[0]:cells(l) for l in raw.splitlines() if l.startswith('| -') and '$_$' in l}
 rows=[cells(l) for l in m[0].splitlines() if l.startswith('| -') and '$_$' in l]
 assert len(rows)==len(old)==100;seen=set();sums=[0,0,0]
 for row in rows:
  sid=row[0].rstrip('*');assert sid not in seen;seen.add(sid);src=old[sid]
  assert row[1:3]==[f'{float(x):.3f}' for x in src[1:3]],sid
  assert row[3]=='/'.join(x.split('x')[0] for x in src[3:7]),sid
  assert row[4]=='/'.join(src[9:12]),sid
  assert row[0].endswith('*')==(src[8]=='是'),sid
  n=[int(x) for x in row[4].split('/')];assert sum(n)==int(src[3].split('x')[0]),sid
  sums=[a+b for a,b in zip(sums,n)]
 assert sums==[1397,71,464]
 q2dir=ROOT/'问题2/docs/final'
 q2rows=[cells(l) for l in m[1].splitlines() if re.match(r'^\| 附件3_\d\d \|',l)]
 with (q2dir/'attachment3_predictions.csv').open(encoding='utf-8-sig') as f:q2pred=list(csv.DictReader(f))
 expected2=[[r['sample_id'].split('.pkl')[0],['负向','中性','正向'][int(r['class_id'])],f"{float(r['intensity']):.4f}",f"{max(float(r['prob_'+c]) for c in ['negative','neutral','positive']):.4f}"] for r in q2pred]
 assert q2rows==expected2 and len(q2rows)==30 and len({r[0] for r in q2rows})==30
 q2metrics=json.loads((q2dir/'validation_metrics.json').read_text());clean=q2metrics['clean']
 assert len([v for v in q2metrics.values() if 'n' in v])==46 and clean['n']==728
 assert table(m[1],'表18')==[[f'{clean[k]:.4f}' for k in ['accuracy','macro_f1','mae','pearson']]]
 for row in table(m[1],'表10'):
  modality={'文本':'text','语音':'audio','视觉':'vision'}[row[0]]
  result=q2metrics[f"{modality}_{row[1].rstrip('%')}_random"]
  assert row[2:]==[f"{result['macro_f1']:.4f}",f"{result['macro_f1']-clean['macro_f1']:+.4f}",f"{result['mae']:.4f}",f"{result['mae']-clean['mae']:+.4f}"]
 for row in table(m[1],'表19'):
  ratio=row[0].rstrip('%');single=q2metrics[f'text_{ratio}_random'];multiple=q2metrics[f'text_{ratio}_multi']
  assert row[1:]==[f"{multiple[k]-single[k]:+.4f}" for k in ['macro_f1','mae']]
 for name,value in json.loads((PAPER/'references/q2-update/manifest.json').read_text())['files'].items():assert digest(ROOT/name)==value,name
 with (Q3/'results/附件4_预测与解释汇总.csv').open(encoding='utf-8-sig') as f:actual={r['id']:r for r in csv.DictReader(f)}
 preds=table(m[2],'表15');locations=table(m[2],'表16');assert len(preds)==len(locations)==len(actual)==20
 for row in preds:
  src=actual[row[0]];assert row[1]==src['predicted_label']
  assert abs(float(row[2])-float(src['predicted_intensity']))<=.00005001
  assert row[3:6]==[f"{100*float(src['classification_modality_shares_'+k]):.2f}" for k in 'TAV']
  assert row[6]==src['classification_principal_modality']+'/'+src['regression_principal_modality']
 for row in locations:
  assert len(row)==5,row
  first=actual[row[0]]['key_evidence_T'].split('\n')[0];frame=re.search(r'同步画面([\d.]+)s',first)
  assert row[4]==(frame[1] if frame else '—')
  score=re.search(r'分数([+\-\d.]+)',first);assert row[2]==score[1]
  if frame:
   bounds=re.search(r'([\d.]+)—([\d.]+)',row[3]);assert bounds and float(bounds[1])<=float(row[4])<=float(bounds[2])
 metrics=json.loads((Q3/'results/uncertainty/metrics.json').read_text())
 for split,cap in [('valid','表4'),('test','表5')]:
  row=table(m[2],cap)[-1];a=metrics[split]['final']
  assert row[1:5]==[f"{100*a['accuracy']:.2f}",*[f'{a[k]:.4f}' for k in ['macro_f1','mae','pearson']]]
 media=json.loads((Q3/'results/evidence_navigation/media_consistency.json').read_text())
 assert [r['id'] for r in media['samples'] if r['feature_availability']['all_zero']['aligned']['vision']]==['13']
 return {'q1_rows':100,'q1_word_states':sums,'q2_rows':30,'q3_prediction_rows':20,'q3_location_rows':20,'q3_metrics_match_json':True,'q2_final_metrics_match_json':True,'q2_final_predictions_match_csv':True,'q2_source_hashes_verified':True,'q2_comparison_scope':'final version q2-final-span-seed42-v1; supplied PDF and recorded CSV/JSON checked; no local model inference'}
def main():
 pdf=PAPER/'paper.pdf';assert pdf.is_file() and pdf.stat().st_size>10000
 info=subprocess.check_output(['pdfinfo',str(pdf)],text=True);pages=int(re.search(r'^Pages:\s+(\d+)',info,re.M)[1])
 text=subprocess.check_output(['pdftotext','-layout',str(pdf),'-'],text=True);sheets=text.split('\f');sheets=[x for x in sheets if x.strip()]
 assert len(sheets)==pages
 assert '学校' not in re.sub(r'\s','', ''.join(sheets[1:]))
 assert '摘要' in sheets[1] and '问题重述与分析' in sheets[2]
 for index,sheet in enumerate(sheets[1:],1):assert sheet.strip().splitlines()[-1].strip()==str(index),(index,'page footer')
 for title in ['问题一','问题二','问题三','参考文献']:assert title in text,title
 for value in ['1397','1468','0.6163','0.6493','0.6006','0.5397','75.98']:assert value in text,value
 assert '13条对齐视觉' not in text
 log=(PAPER/'build/main.log').read_text(errors='replace')
 fatal=[l for l in log.splitlines() if any(x in l for x in ['Missing character:','undefined references','Overfull \\hbox','Overfull \\vbox','already defined']) or ('Citation' in l and 'undefined' in l)]
 assert not fatal,'\n'.join(fatal)
 sources=json.loads((PAPER/'sources.json').read_text())
 for r in sources['sources']:
  assert digest(ROOT/r['source'])==r['source_sha256'];assert digest(ROOT/r['research_source'])==r['research_sha256']
  assert digest(PAPER/'chapters'/f"question{r['question']}.tex")==r['chapter_sha256']
 for n,v in sources['paper_inputs'].items():assert digest(PAPER/n)==v,n
 assert digest(PAPER/'scripts/sync_chapters.py')==sources['generator_sha256']
 assert digest(Q3/'paper/build_paper.py')==sources['q3_equation_generator_sha256']
 for n,r in sources['assets'].items():assert digest(ROOT/n)==r['sha256']
 # The bibliography contains exactly the first-citation order of used sources.
 ordered=[]
 for i in range(1,4):
  for group in re.findall(r'\\cite\{([^}]+)\}',(PAPER/'chapters'/f'question{i}.tex').read_text()):
   for key in group.split(','):
    if key not in ordered:ordered.append(key)
 assert ordered==sources['citation_order']==re.findall(r'\\bibitem\{([^}]+)\}',(PAPER/'chapters/references.tex').read_text())
 for record in json.loads((PAPER/'assets/figure_sources.json').read_text()):
  original=ROOT/record['source'];assert digest(original)==record['source_sha256']
  expected=original.read_text()
  for a,b in [('Round 3 reference','Unweighted'),('Frozen round 4','Class-balanced'),('Round 3','Unweighted'),('Round 4','Class-balanced')]:expected=expected.replace(a,b)
  assert (PAPER/record['svg']).read_text()==expected
  assert digest(PAPER/record['pdf'])==record['pdf_sha256']
 assert not re.search(r'Round [2-6]|Frozen round [2-6]',text)
 results=check_results()
 # Recompute the added postprocessing comparison from saved predictions.
 manuscript=(PAPER/'manuscript/question3.md').read_text()
 postrows=table(manuscript,'表18')
 for row,model in zip(postrows,['selfmm','tetfn','uncertainty'],strict=True):
  with (Q3/'results'/model/'valid_predictions.csv').open(encoding='utf-8-sig') as f: predictions=list(csv.DictReader(f))
  assert len(predictions)==728
  mae=lambda column:sum(abs(float(r[column])-float(r['true_intensity'])) for r in predictions)/len(predictions)
  changed=sum(abs(float(r['intensity'])-float(r['raw_intensity']))>1e-9 for r in predictions)
  assert row[1:]==[f"{mae('raw_intensity'):.4f}",f"{mae('intensity'):.4f}",f'{changed}/728']
 for n in range(1,4):
  manuscript=(PAPER/'manuscript'/f'question{n}.md').read_text()
  for heading in re.finditer(r'^(#{2,3}) ([^\n]+)\n\s*\n(#{2,3}) ',manuscript,re.M):
   assert len(heading[3])>len(heading[1]),('empty heading',n,heading[2])
 results['postprocessing_table_matches_csv']=True
 report={'status':'passed','pages':pages,'pdf_bytes':pdf.stat().st_size,'pdf_sha256':digest(pdf),'sources_verified':True,'abstract_pages':1,'continuous_page_numbers':True,'undefined_references':False,'missing_glyphs':False,'overfull_boxes':False,'result_checks':results,'scope':'PDF compilation, pagination, source hashes, tables and recorded metrics; no new model training or independent accuracy evaluation.'}
 (PAPER/'build/verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
