"""Check original media identity and token differences between released versions."""
import argparse,collections,json,pickle,subprocess,hashlib
from pathlib import Path
import numpy as np
from transformers import BertTokenizerFast
from check_attachments import sha,write,spans,load

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('/workspace/data'));p.add_argument('--tokenizer',type=Path,default=Path('/workspace/q1-models/text'));p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent);a=p.parse_args()
 tokenizer=BertTokenizerFast.from_pretrained(str(a.tokenizer),local_files_only=True)
 q3=a.data/'附件3-模态缺失特征样本';q4=next((a.data/'附件4-可解释专项视频样本与特征文件').glob('附件4*'));out={'attachment3_tokens':[],'attachment4_versions':[],'media':[]}
 for p in sorted((q3/'对齐版本').glob('*.pkl')):
  num=p.stem.rsplit('_',1)[-1];x=load(p)['test'];u=load(q3/'未对齐版本'/f'附件3_未对齐版本_{num}.pkl')['test'];text=str(u['raw_text'][0]);tok=tokenizer(text,padding='max_length',truncation=True,max_length=50);expected=np.array([tok[k] for k in ['input_ids','attention_mask','token_type_ids']]);actual=x['text_bert'][0];changes=np.argwhere(expected!=actual)
  row={'number':num,'text_from_unaligned_version':text,'token_changes':[{'channel':int(i),'position':int(j),'expected_from_raw_text':int(expected[i,j]),'released':int(actual[i,j])} for i,j in changes],'unk_positions':np.flatnonzero(actual[0]==100).tolist(),'zero_rows':{k:spans(np.all(x[k][0]==0,axis=-1)) for k in ['audio','vision']},'unknown_token':tokenizer.convert_ids_to_tokens(100)}
  indices=np.flatnonzero(actual[0]==100)
  row['unk_and_audio_zero_positions']=[int(j) for j in indices if np.all(x['audio'][0,j]==0)]
  row['unk_and_vision_zero_positions']=[int(j) for j in indices if np.all(x['vision'][0,j]==0)]
  row['unk_with_attention_one_positions']=[int(j) for j in indices if actual[1,j]==1]
  out['attachment3_tokens'].append(row)
 for p in sorted((q4/'对齐版本').glob('*.pkl')):
  x=load(p);u=load(q4/'未对齐版本'/p.name)
  out['attachment4_versions'].append({'number':p.stem,'shared_fields_equal':{k:bool(np.array_equal(x[k],u[k])) for k in ['raw_text','text','text_bert']},'all_zero':{v:{k:bool(np.all(d[k]==0)) for k in ['audio','vision']} for v,d in [('aligned',x),('unaligned',u)]},'nonzero_rows':{v:{k:int(np.any(d[k]!=0,axis=-1).sum()) for k in ['audio','vision']} for v,d in [('aligned',x),('unaligned',u)]}})
 files=[(1,p.parent.name+'$_$'+p.stem,p) for p in sorted((a.data/'附件1-数据集原始多模态样本').rglob('*.mp4'))]+[(4,p.stem,p) for p in sorted((q4/'对齐版本/videos').glob('*.mp4'))]
 for attachment,sid,p in files:
  probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(p)]))
  raw=subprocess.check_output(['ffmpeg','-nostdin','-v','error','-i',str(p),'-map','0:a:0','-ac','1','-ar','16000','-f','f32le','-']);wave=np.frombuffer(raw,dtype='<f4')
  v=next(s for s in probe['streams'] if s['codec_type']=='video');audio=next(s for s in probe['streams'] if s['codec_type']=='audio')
  row={'attachment':attachment,'id':sid,'path':str(p.relative_to(a.data)),'sha256':sha(p),'pcm_mono_16k_sha256':hashlib.sha256(raw).hexdigest(),'zero_audio':bool(not np.any(wave)),'pcm_duration_s':len(wave)/16000,'container_duration_s':float(probe['format']['duration']),'audio_stream_duration_s':float(audio.get('duration',0)),'video_stream_duration_s':float(v.get('duration',0))}
  if attachment==4:
   other=q4/'未对齐版本/videos'/p.name
   row['versions_video_byte_equal']=other.exists() and sha(other)==row['sha256']
  out['media'].append(row)
  if len(out['media'])%20==0:print('media',len(out['media']),flush=True)
 for kind in ['sha256','pcm_mono_16k_sha256']:
  groups=collections.defaultdict(list)
  for r in out['media']:groups[r[kind]].append({'attachment':r['attachment'],'id':r['id']})
  out[kind+'_duplicate_groups']=[v for v in groups.values() if len(v)>1]
 out['script_sha256']=sha(Path(__file__));out['tokenizer_vocab_sha256']=sha(a.tokenizer/'vocab.txt');write(a.output/'media_and_token_comparison.json',out)
if __name__=='__main__':main()
