# 本程序及代码是在人工智能工具辅助下完成的。
# 工具：OpenAI Codex；模型：GPT-6 Astra (gpt-6-astra)；开发机构：OpenAI；模型发布日期：2026-09-03。
# AI 来源说明于 2026-09-24 补录，未改变算法；参见 AI辅助使用记录与队内审阅要求.md。
"""Independent ASR diagnostics only: never replace official text or labels."""
import argparse, dataclasses, json, time
from pathlib import Path
import numpy as np
from faster_whisper import WhisperModel
from q1_extract import command, sha, dump, normalize

OUT=Path('results/q1_next'); BASE=Path('results/q1/samples')
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--limit',type=int); args=ap.parse_args()
    lock=json.loads((OUT/'asr_model.lock.json').read_text())
    for name,digest in lock['files'].items(): assert sha(Path(lock['path'])/name)==digest
    model=WhisperModel(lock['path'],device='cuda',compute_type='float16',cpu_threads=8,local_files_only=True)
    config=dict(beam_size=5,temperature=0,condition_on_previous_text=False,word_timestamps=True,
                vad_filter=True,vad_parameters=dict(threshold=.5,min_silence_duration_ms=500,speech_pad_ms=200))
    dump(OUT/'asr_config.json',dict(model=lock['repo'],revision=lock['revision'],**config,
         official_transcript_used_as_prompt=False,labels_loaded=False,role='auxiliary diagnostics only'))
    dest=OUT/'asr';dest.mkdir(exist_ok=True)
    files=sorted(BASE.glob('*.json'),key=lambda p:json.loads(p.read_text())['table_row_0based'])
    for p in files[:args.limit]:
        output=dest/p.name
        if output.exists(): continue
        meta=json.loads(p.read_text()); source=Path('/data')/meta['source_path']; assert sha(source)==meta['source_sha256']
        wave=np.frombuffer(command(['ffmpeg','-v','error','-i',source,'-map','0:a:0','-ac','1','-ar','16000','-f','f32le','pipe:1']),dtype='<f4').copy()
        start=time.time(); segments,info=model.transcribe(wave,**config); segments=[dataclasses.asdict(s) for s in segments]
        result=dict(sample_id=meta['sample_id'],source_sha256=meta['source_sha256'],language=info.language,
                    language_probability=info.language_probability,duration=info.duration,
                    duration_after_vad=info.duration_after_vad,segments=segments,
                    transcript=' '.join(s['text'].strip() for s in segments),audio_start_s=meta['audio_start_s'])
        if info.language!='en' and segments:
            translated,_=model.transcribe(wave,language=info.language,task='translate',**{**config,'word_timestamps':False})
            result['translation_segments']=[dataclasses.asdict(s) for s in translated]
            result['translation']=' '.join(s['text'].strip() for s in result['translation_segments'])
        result['elapsed_seconds']=round(time.time()-start,3);dump(output,result)
        print(json.dumps({k:result[k] for k in ('sample_id','language','language_probability','duration_after_vad','transcript','elapsed_seconds')},ensure_ascii=False),flush=True)
if __name__=='__main__': main()
