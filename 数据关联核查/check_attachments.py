"""Read-only cross-attachment identity and observed corruption audit.
Special-set labels are neither joined nor exported. Equality is checked at the
released target dtype so float64 -> float32 serialization is not corruption.
"""
import argparse,collections,hashlib,json,pickle,re,subprocess,zipfile
from pathlib import Path
import numpy as np

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def load(p):
 with p.open('rb') as f:return pickle.load(f)
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def spans(mask):
 ix=np.flatnonzero(mask);out=[]
 for i in ix:
  if out and out[-1][1]==i:out[-1][1]=int(i+1)
  else:out.append([int(i),int(i+1)])
 return out

def difference(source,target):
 a=np.asarray(source).astype(np.asarray(target).dtype);b=np.asarray(target)
 assert a.shape==b.shape
 original_shape=list(b.shape)
 if b.shape==(3,50):a,b=a.T,b.T
 eq=(a==b)|(np.isnan(a)&np.isnan(b)) if a.dtype.kind=='f' else a==b
 changed=~eq
 zr=np.all(b==0,axis=-1);sr=np.all(a==0,axis=-1)
 return {'source_dtype':str(np.asarray(source).dtype),'target_dtype':str(b.dtype),'shape':original_shape,'equal_after_target_cast':bool(eq.all()),'changed_cells':int(changed.sum()),'changed_nonzero_target_cells':int((changed&(b!=0)).sum()),'new_zero_cells':int((changed&(b==0)).sum()),'new_zero_row_spans':spans(zr&~sr),'source_zero_row_spans':spans(sr),'target_zero_row_spans':spans(zr),'nonfinite_target_cells':int((~np.isfinite(b)).sum())}

def bank(data):
 refs=[(split,i,str(sid)) for split,d in data.items() for i,sid in enumerate(d['id'])]
 fields={k:np.concatenate([d[k] for d in data.values()],axis=0) for k in ['text_bert','audio','vision','text','raw_text']}
 return refs,fields

def match_field(values,target):
 target=np.asarray(target)
 if target.shape==(3,50):values,target=values.transpose(0,2,1),target.T
 observed=np.flatnonzero(np.any(target!=0,axis=-1));candidates=np.arange(len(values))
 for pos in observed:
  candidates=candidates[np.all((values[candidates,pos].astype(target.dtype)==target[pos])|(target[pos]==0),axis=-1)]
  if not len(candidates):break
 return candidates.tolist() if len(observed) else [],len(observed)

def source_meta(ref):return {'split':ref[0],'row':ref[1],'source_id':ref[2]}

def audit_version(data_root,version):
 source=data_root/'附件2-数据集特征文件'/('aligned_50.pkl' if version=='对齐版本' else 'unaligned_50.pkl')
 data=load(source);refs,b=bank(data);ids=[r[2] for r in refs];byid={sid:i for i,sid in enumerate(ids)}
 text_index=collections.defaultdict(list)
 for i,t in enumerate(b['raw_text']):text_index[str(t)].append(i)
 summary={'samples':len(ids),'splits':{s:len(d['id']) for s,d in data.items()},'duplicate_ids':[k for k,v in collections.Counter(ids).items() if v>1],'source_sha256':sha(source),'all_zero_samples':{},'nonfinite':{}}
 for key in ['text','text_bert','audio','vision']:
  summary['all_zero_samples'][key]=[ids[i] for i in np.flatnonzero(np.all(b[key]==0,axis=(1,2)))];summary['nonfinite'][key]=int((~np.isfinite(b[key])).sum())
 rows3=[];rows4=[]
 for p in sorted((data_root/'附件3-模态缺失特征样本'/version).glob('*.pkl')):
  x=load(p)['test'];x={k:np.asarray(v)[0] for k,v in x.items()};candidate_sets=[];fieldmatches={}
  if 'raw_text' in x:
   matches=text_index.get(str(x['raw_text']),[]);fieldmatches['raw_text']=matches
   if matches:candidate_sets.append(set(matches))
  for key in ['text_bert','audio','vision']:
   if key not in x:continue
   matches,observed=match_field(b[key],x[key]);fieldmatches[key]={'matches':[source_meta(refs[i]) for i in matches],'observed_rows':observed}
   if observed:candidate_sets.append(set(matches))
  common=set.intersection(*candidate_sets) if candidate_sets else set()
  if not common:
   votes=collections.Counter(i for options in candidate_sets for i in options)
   common={i for i,count in votes.items() if count>=2}
  row={'file':str(p.relative_to(data_root)),'sha256':sha(p),'number':p.stem.rsplit('_',1)[-1],'candidates':[source_meta(refs[i]) for i in sorted(common)],'field_matches':fieldmatches}
  if len(common)==1:
   i=next(iter(common));row['source']=source_meta(refs[i]);row['differences']={k:difference(b[k][i],x[k]) for k in ['text_bert','audio','vision'] if k in x}
   if version=='未对齐版本':row['source_lengths']={k:int(data[refs[i][0]][k][refs[i][1]]) for k in ['audio_lengths','vision_lengths']}
  rows3.append(row)
 base=next((data_root/'附件4-可解释专项视频样本与特征文件').glob('附件4*'))/version
 for p in sorted(base.glob('*.pkl')):
  x=load(p);matches=text_index.get(str(x['raw_text']),[])
  # Locate by raw text, then verify every numeric field independently.
  row={'number':p.stem,'file':str(p.relative_to(data_root)),'sha256':sha(p),'text_candidates':[source_meta(refs[i]) for i in matches]}
  if len(matches)==1:
   i=matches[0];row['source']=source_meta(refs[i]);row['differences']={k:difference(b[k][i],x[k]) for k in ['text','text_bert','audio','vision']}
   if version=='未对齐版本':row['length_match']={k:bool(x[k]==data[refs[i][0]][k][refs[i][1]]) for k in ['audio_lengths','vision_lengths']}
  else:
   row['feature_candidates']={k:[source_meta(refs[i]) for i in match_field(b[k],x[k])[0]] for k in ['text_bert','audio','vision']}
  rows4.append(row)
 print(version,'A3 mapped',sum('source' in r for r in rows3),'A4 mapped',sum('source' in r for r in rows4),flush=True)
 return {'attachment2':summary,'attachment3':rows3,'attachment4':rows4},refs,b,data

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('/workspace/data'));p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 out={'scope':'Original attachment files; no special-set labels used or changed','versions':{}}
 aligned,refs,b,adata=audit_version(a.data,'对齐版本');out['versions']['aligned']=aligned
 # Only identity and fields needed for version correspondence persist.
 common_fields={sid:{'raw_text':str(b['raw_text'][i]),'text_bert':b['text_bert'][i].copy(),'text':b['text'][i].copy()} for i,(_,_,sid) in enumerate(refs)}
 q1_path=next((a.data/'附件1-数据集原始多模态样本').rglob('label-100.xlsx'))
 import openpyxl
 wb=openpyxl.load_workbook(q1_path,read_only=True,data_only=True);rows=list(wb.active.values);header=rows[0];q1=[]
 for values in rows[1:]:
  d=dict(zip(header,values));sid=str(d['video_id'])+'$_$'+str(int(d['clip_id']));indices=[i for i,ref in enumerate(refs) if ref[2]==sid]
  q1.append({'source_id':sid,'attachment2_matches':[source_meta(refs[i]) for i in indices],'text_equal':str(d['text'])==str(b['raw_text'][indices[0]]) if len(indices)==1 else None,'attachment1_text':str(d['text']),'attachment2_text':str(b['raw_text'][indices[0]]) if len(indices)==1 else None})
 out['attachment1_to_2']=q1;out['attachment1_label_file_sha256']=sha(q1_path)
 del b,adata
 unaligned,urefs,ub,udata=audit_version(a.data,'未对齐版本');out['versions']['unaligned']=unaligned
 diffs=[]
 for i,(split,row,sid) in enumerate(urefs):
  if sid not in common_fields:diffs.append({'id':sid,'kind':'id_missing_aligned'});continue
  prior=common_fields[sid]
  for k in ['raw_text','text_bert','text']:
   if not np.array_equal(prior[k],ub[k][i]):diffs.append({'id':sid,'kind':k+'_differs'})
 out['attachment2_cross_version']={'same_id_set':set(common_fields)=={r[2] for r in urefs},'same_id_split_sequence':refs==urefs,'text_field_differences':diffs}
 for att in ['attachment3','attachment4']:
  av={r['number']:r for r in aligned[att]};uv={r['number']:r for r in unaligned[att]}
  out[att+'_cross_version']=[{'number':n,'aligned_source':av[n].get('source'),'unaligned_source':uv[n].get('source'),'same_source':av[n].get('source',{}).get('source_id')==uv[n].get('source',{}).get('source_id') if 'source' in av[n] and 'source' in uv[n] else None} for n in sorted(av)]
 out['script_sha256']=sha(Path(__file__));write(a.output/'feature_comparison.json',out)
 print('saved',a.output/'feature_comparison.json',flush=True)
if __name__=='__main__':main()
