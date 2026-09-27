"""Verify immutable token/character identity and inventory source media clocks."""
import argparse, json, pickle, subprocess
from pathlib import Path
import numpy as np
from transformers import BertTokenizerFast
from common import read, write, digest
ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    tokenizer=BertTokenizerFast.from_pretrained(str(a.tokenizer),local_files_only=True)
    mappings=[json.loads(l) for l in (ROOT/'results/media_mapping.jsonl').read_text().splitlines()]
    samples=[]
    for m in mappings:
        sid=m['id'];src=a.source/(sid+'.pkl');x=pickle.loads(src.read_bytes());text=str(x['raw_text'])
        assert text==m['raw_text']
        token=tokenizer(text,truncation=True,max_length=50,padding='max_length',return_offsets_mapping=True)
        ids=np.asarray(x['text_bert']);assert np.array_equal(ids,np.asarray([token[k] for k in ['input_ids','attention_mask','token_type_ids']]))
        for e in m['entries']:
            for j in e['source_indices']:
                assert int(ids[0,j])==e['token_id'] and list(token['offset_mapping'][j]) in e['char_spans']
        video=a.source/'videos'/(sid+'.mp4');assert digest(video)==m['provenance']['video_sha256']
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(video)]))
        audio=next(s for s in probe['streams'] if s['codec_type']=='audio');origin=float(audio.get('start_time',0))
        raw=subprocess.check_output(['ffmpeg','-nostdin','-v','error','-i',str(video),'-map','0:a:0','-ac','1','-ar','16000','-f','f32le','-'])
        wave=np.frombuffer(raw,dtype='<f4')
        frames=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_entries','frame=best_effort_timestamp_time','-of','json',str(video)]))['frames']
        pts=[float(f['best_effort_timestamp_time']) for f in frames]
        assert pts and all(b>a for a,b in zip(pts,pts[1:]))
        samples.append({'sample_id':sid,'path':sid+'.mp4','sha256':digest(video),'feature_sha256':digest(src),'raw_text':text,'audio':{'start_s':origin,'end_s':origin+len(wave)/16000,'samples_16k':len(wave)},'zero_audio':not bool(np.any(wave)),'video_frame_pts_s':pts,'token_character_identity_verified':True,'source_feature_timestamps_available':False})
    write(a.output,{'source_root':str(a.source/'videos'),'samples':samples,'mapping_sha256':digest(ROOT/'results/media_mapping.jsonl'),'tokenizer_vocab_sha256':digest(a.tokenizer/'vocab.txt')})
    print({'verified_samples':len(samples)})
if __name__=='__main__':main()
