"""Check supplied label tables and raw-text/token identity within attachment 2."""
import argparse,json
from pathlib import Path
import numpy as np,openpyxl
from transformers import BertTokenizerFast
from check_attachments import load,sha,write

def table(path):
 w=openpyxl.load_workbook(path,read_only=True,data_only=True);r=list(w['label'].values);return [dict(zip(r[0],row)) for row in r[1:]],w

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('/workspace/data'));p.add_argument('--tokenizer',type=Path,default=Path('/workspace/q1-models/text'));p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent);a=p.parse_args()
 f=a.data/'附件2-数据集特征文件';source=load(f/'aligned_50.pkl');rows,_=table(f/'label.xlsx');byid={str(r['video_id'])+'$_$'+str(int(r['clip_id'])):r for r in rows};tokenizer=BertTokenizerFast.from_pretrained(a.tokenizer,local_files_only=True);issues=[];token_issues=[];all_labels={}
 for split,d in source.items():
  tok=tokenizer(list(map(str,d['raw_text'])),padding='max_length',truncation=True,max_length=50);expected=np.stack([tok[k] for k in ['input_ids','attention_mask','token_type_ids']],axis=1)
  for i,sid in enumerate(d['id']):
   sid=str(sid);r=byid.get(sid);y=float(d['regression_labels'][i]);cls=int(d['classification_labels'][i]);all_labels[sid]=y
   if r is None:issues.append({'id':sid,'reason':'spreadsheet_id_missing'});continue
   checks={'text':str(r['text'])==str(d['raw_text'][i]),'split':r['mode']==split,'intensity':abs(float(r['label'])-y)<=1e-7,'class':cls==(0 if y<0 else 2 if y>0 else 1),'annotation':r['annotation']==('Negative' if y<0 else 'Positive' if y>0 else 'Neutral')}
   issues.extend({'id':sid,'reason':k} for k,v in checks.items() if not v)
   if not np.array_equal(expected[i],d['text_bert'][i]):token_issues.append({'id':sid,'changed_cells':int(np.count_nonzero(expected[i]!=d['text_bert'][i]))})
 unaligned=load(f/'unaligned_50.pkl')
 label_versions={split:{k:bool(np.array_equal(d[k],unaligned[split][k])) for k in ['classification_labels','regression_labels']} for split,d in source.items()}
 a1file=next((a.data/'附件1-数据集原始多模态样本').rglob('label-100.xlsx'));r1,w1=table(a1file);q1issues=[];shared=0
 for r in r1:
  sid=str(r['video_id'])+'$_$'+str(int(r['clip_id']));y=float(r['label']);expected='Negative' if y<0 else 'Positive' if y>0 else 'Neutral'
  if r['annotation']!=expected:q1issues.append({'id':sid,'reason':'annotation_sign'})
  if sid in all_labels:
   shared+=1
   if abs(all_labels[sid]-y)>1e-7:q1issues.append({'id':sid,'reason':'cross_attachment_intensity'})
 notes=[list(row) for row in w1['修复说明'].values if row[0]!='源文件'] if '修复说明' in w1.sheetnames else []
 write(a.output/'label_and_text_checks.json',{'attachment2_samples':len(all_labels),'attachment2_spreadsheet_sha256':sha(f/'label.xlsx'),'attachment2_spreadsheet_rows':len(rows),'attachment2_label_versions_equal':label_versions,'attachment2_label_text_split_issues':issues,'attachment2_tokenization_issues':token_issues,'attachment1_samples':len(r1),'attachment1_shared_labels_compared':shared,'attachment1_label_issues':q1issues,'attachment1_repair_note':notes,'special_labels_exported':False,'script_sha256':sha(Path(__file__))});print('label issues',len(issues),len(q1issues),'token issues',len(token_issues))
if __name__=='__main__':main()
