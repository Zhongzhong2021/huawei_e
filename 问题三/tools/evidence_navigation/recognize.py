"""Unprompted full-corpus ASR. Reference text and human records are not model inputs."""
import argparse
import importlib.metadata
import subprocess
import time
from pathlib import Path
import numpy as np
from faster_whisper import WhisperModel
from common import read, write, digest

ROOT = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--models', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source', type=Path)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--condition', choices=['identity', 'half_gain', 'leading_silence_1s'], default='identity')
    args = p.parse_args()
    cfg = read(ROOT / 'config/whisper_alignment.json')
    inputs = read(args.inputs)
    manifest = read(args.models / 'manifest.json')
    assert manifest['revision'] == cfg['revision']
    for f in manifest['files']:
        assert digest(args.models / f['path']) == f['sha256']
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {'config': cfg, 'config_sha256': digest(ROOT / 'config/whisper_alignment.json'),
                'script_sha256': digest(__file__), 'input_manifest_sha256': digest(args.inputs),
                'model': manifest, 'device': args.device,
                'compute_type': cfg['compute_type'] if args.device == 'cuda' else 'int8',
                'condition': args.condition, 'reference_prompt_used': False,
                'human_data_used': False, 'label_used': False,
                'versions': {n: importlib.metadata.version(n) for n in ['faster-whisper','ctranslate2','numpy','av']}}
    write(args.output / 'protocol.json', protocol)
    model = WhisperModel(str(args.models), device=args.device, compute_type=protocol['compute_type'], cpu_threads=8, num_workers=1, local_files_only=True)
    records = []
    for index, item in enumerate(inputs['samples']):
        start = time.monotonic()
        path = (args.source or Path(inputs['source_root'])) / item['path']
        assert digest(path) == item['sha256']
        # Decode the same actual audio support; ffmpeg resamples without VAD or editing.
        raw = subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(path),'-map','0:a:0','-ac','1','-ar','16000','-f','f32le','-'], check=True, capture_output=True).stdout
        wave = np.frombuffer(raw, dtype='<f4').copy()
        shift = 1.0 if args.condition == 'leading_silence_1s' else 0.0
        if args.condition == 'half_gain': wave *= 0.5
        if shift: wave = np.concatenate([np.zeros(16000, dtype=np.float32), wave])
        segments, words = [], []
        if np.any(wave):
            stream, info = model.transcribe(wave, **cfg['recognition'])
            for seg in stream:
                segments.append({'text': seg.text, 'start': seg.start, 'end': seg.end,
                                 'avg_logprob': seg.avg_logprob, 'no_speech_prob': seg.no_speech_prob})
                for w in seg.words or []:
                    words.append({'text': w.word, 'interval_s': [item['audio']['start_s']+w.start, item['audio']['start_s']+w.end], 'probability': w.probability})
        result = {'sample_id': item['sample_id'], 'source_sha256': item['sha256'],
                  'condition': args.condition, 'source_time_shift_s': shift,
                  'zero_audio': not bool(np.any(wave)), 'segments': segments, 'words': words,
                  'elapsed_s': time.monotonic()-start}
        name = item['sample_id'] + '.json'
        write(args.output / name, result)
        records.append({'sample_id': item['sample_id'], 'sha256': digest(args.output/name), 'words': len(words)})
        print(index+1, item['sample_id'], len(words), round(result['elapsed_s'],2), flush=True)
    write(args.output / 'manifest.json', {'complete': True, 'samples': records,
          'protocol_sha256': digest(args.output / 'protocol.json'), 'human_data_used': False})

if __name__ == '__main__':
    main()
