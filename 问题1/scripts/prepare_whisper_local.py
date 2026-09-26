"""Prepare two fixed-context phonetic alignment crops per supported ASR run."""
import argparse
import math
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from common import read, write, digest, run
from whisper_support import lexical_support

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--recognition', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--source', type=Path)
    args = p.parse_args()
    manifest = read(args.recognition/'manifest.json')
    assert manifest['complete'] and len(manifest['samples']) == 100
    assert digest(args.recognition/'protocol.json') == manifest['protocol_sha256']
    assert read(args.recognition/'protocol.json')['condition'] == 'identity'
    cfg = read(ROOT/'config/automatic_alignment.json')
    inputs = read(ROOT/'data/inputs.json')
    args.output.mkdir(parents=True, exist_ok=False)
    corpus = args.work/'corpus'; corpus.mkdir(parents=True, exist_ok=False)
    rows, jobs = [], []
    files = {r['sample_id']:r['sha256'] for r in manifest['samples']}
    for source in inputs['samples']:
        sid = source['sample_id']
        meta = read(ROOT/'data/features'/(sid+'.json'))
        path = args.recognition/(sid+'.json')
        assert digest(path) == files[sid]
        recognized = read(path)
        assert recognized['source_sha256'] == source['sha256']
        support = lexical_support(meta['words'], recognized['words'], cfg, source['audio']['start_s'], source['audio']['end_s'])
        row = {'sample_id':sid, 'source_sha256':source['sha256'], 'recognition_sha256':digest(path),
               'metadata_sha256':digest(ROOT/'data/features'/(sid+'.json')), 'support':support, 'jobs':[]}
        if source['zero_audio']:
            assert not support['runs']
        if support['runs']:
            path = (args.source or Path(inputs['source_root']))/source['path']
            assert digest(path) == source['sha256']
            wave = np.frombuffer(run(['ffmpeg','-nostdin','-v','error','-i',str(path),'-map','0:a:0','-ac','1','-ar','16000','-f','f32le','-']), '<f4')
            origin = source['audio']['start_s']
            for run_index, proposal in enumerate(support['runs']):
                indices = proposal['word_indices']
                for context in [0.5, 1.0]:
                    lo = max(0, math.floor((proposal['interval_s'][0]-origin-context)*16000))
                    hi = min(len(wave), math.ceil((proposal['interval_s'][1]-origin+context)*16000))
                    assert hi > lo
                    job_id = f'w{len(jobs):05d}'
                    wav, lab = corpus/(job_id+'.wav'), corpus/(job_id+'.lab')
                    wavfile.write(wav, 16000, wave[lo:hi].copy())
                    lab.write_text(' '.join(p for i in indices for p in meta['words'][i]['parts']).lower()+'\n')
                    job = {'file_id':job_id,'sample_id':sid,'run_index':run_index,'word_indices':indices,
                           'source_words':[meta['words'][i] for i in indices], 'context_s':context,
                           'sample_span_16k':[lo,hi],'source_offset_s':origin+lo/16000,'duration_s':(hi-lo)/16000,
                           'wav_sha256':digest(wav),'text_sha256':digest(lab)}
                    jobs.append(job);row['jobs'].append(job_id)
        rows.append(row)
    write(args.output/'preparation.json', {'schema':'q1-whisper-local-v1','samples':rows,'jobs':jobs,
           'contexts_s':[0.5,1.0],'human_data_used':False,'recognition_manifest_sha256':digest(args.recognition/'manifest.json'),
           'baseline_config_sha256':digest(ROOT/'config/automatic_alignment.json'),
           'scripts':{n:digest(ROOT/'scripts'/n) for n in ['whisper_support.py','prepare_whisper_local.py']}})
    print({'samples':len(rows),'crops':len(jobs),'runs':sum(len(r['support']['runs']) for r in rows)})

if __name__ == '__main__':
    main()
