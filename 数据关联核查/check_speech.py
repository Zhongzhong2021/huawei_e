"""Second, unprompted recognizer for all twenty attachment-4 clips."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np,torch
from transformers import AutoProcessor,AutoModelForCTC
from check_attachments import sha,write,load

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('/workspace/data'));p.add_argument('--model',type=Path,default=Path('/workspace/q1-models/ctc'));p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent);a=p.parse_args()
 torch.set_num_threads(4);processor=AutoProcessor.from_pretrained(a.model,local_files_only=True);model=AutoModelForCTC.from_pretrained(a.model,local_files_only=True).eval()
 base=next((a.data/'附件4-可解释专项视频样本与特征文件').glob('附件4*'))/'对齐版本';rows=[]
 for path in sorted((base/'videos').glob('*.mp4')):
  raw=subprocess.check_output(['ffmpeg','-nostdin','-v','error','-i',str(path),'-map','0:a:0','-ac','1','-ar','16000','-f','f32le','-']);wave=np.frombuffer(raw,dtype='<f4').copy();inputs=processor(wave,sampling_rate=16000,return_tensors='pt')
  with torch.inference_mode():tokens=model(**inputs).logits.argmax(-1)
  rows.append({'number':path.stem,'video_sha256':sha(path),'ctc_text':processor.batch_decode(tokens)[0]});print(path.stem,rows[-1]['ctc_text'],flush=True)
 write(a.output/'independent_ctc.json',{'reference_prompt_used':False,'model_files_sha256':{p.name:sha(p) for p in a.model.iterdir() if p.is_file()},'samples':rows,'script_sha256':sha(Path(__file__))})
if __name__=='__main__':main()
