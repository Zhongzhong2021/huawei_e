"""Full-dataset observability and frozen signal-transformation experiments.

No manual labels/times or emotion labels enter generation or comparisons.
Temporal stability is relative to the original automatic result, not truth.
"""
import argparse
import copy
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torchaudio
from common import read, write, digest, run
from extract import Engine
from math_features import cells, acoustic
from feature_io import load, pad
from run_automatic_alignment import decode_words
from selective_alignment import align_sample
from local_evidence import corroborated_local_candidates

ROOT = Path(__file__).resolve().parents[1]


def distribution(values):
    a = np.asarray(values, dtype=float)
    return {'n': len(a), 'median': float(np.median(a)) if len(a) else None,
            'p95': float(np.quantile(a, .95)) if len(a) else None,
            'max': float(a.max()) if len(a) else None}


@torch.inference_mode()
def infer(engine, wave, words, start, cfg):
    baseline, _, _, _, _ = engine.align(wave, copy.deepcopy(words), start)
    hypothesis = []
    if np.any(wave):
        x = engine.processor(wave, sampling_rate=16000, return_tensors='pt')
        lp = engine.ctc(x.input_values).logits[0].log_softmax(-1)
        stride = field = 1
        for k, s in zip(engine.ctc.config.conv_kernel, engine.ctc.config.conv_stride):
            field += (k - 1) * stride
            stride *= s
        clock = cells(start + (np.arange(len(lp))*stride + (field-1)/2)/16000,
                      start, start + len(wave)/16000)
        hypothesis = decode_words(lp.argmax(-1).numpy(), lp,
                                 engine.processor.tokenizer.get_vocab(),
                                 engine.ctc.config.pad_token_id, clock)
    result = align_sample(words, hypothesis, baseline, cfg, no_signal=not np.any(wave))
    result['recognition'] = hypothesis
    return corroborated_local_candidates(result, cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = read(ROOT/'config/automatic_alignment.json')
    method = read(ROOT/'config/method.json')
    inputs = read(ROOT/'data/inputs.json')
    names = ['extract.py', 'math_features.py', 'ctc_alignment.py', 'local_alignment.py',
             'selective_alignment.py', 'local_evidence.py', 'run_automatic_alignment.py',
             'feature_io.py', 'evaluate_pipeline_quality.py']
    protocol = {
        'status': 'fixed_before_execution', 'samples': 100,
        'variants': {'identity': 'original audio', 'half_gain': 'multiply PCM by 0.5',
                     'leading_silence_1s': 'prepend exactly 16000 zero samples after resampling'},
        'rules': 'Run unchanged recognition and forced-alignment branches, then fixed selective policy. No threshold tuning.',
        'comparison': 'Same original word indices; boundary residual subtracts the known 1s shift. Compare only jointly emitted candidates and report retention separately.',
        'not_measured': ['semantic precision', 'true word-boundary error', 'emotion accuracy'],
        'human_records_used': False, 'emotion_labels_used': False,
        'scripts': {n: digest(ROOT/'scripts'/n) for n in names},
        'config_sha256': digest(ROOT/'config/automatic_alignment.json'),
        'method_sha256': digest(ROOT/'config/method.json'),
        'inputs_sha256': digest(ROOT/'data/inputs.json')}
    write(args.output/'protocol.json', protocol)
    for item in read(args.models/'manifest.json'):
        assert digest(args.models/item['path']) == item['sha256']
    torch.set_num_threads(2)
    torch.manual_seed(0)
    engine = Engine(args.models, method)
    observability, experiments = [], []
    states = Counter()
    for index, source in enumerate(inputs['samples']):
        sid = source['sample_id']
        arrays, meta = load(ROOT/'data/features', sid)
        original = read(ROOT/'data/local_evidence_corroborated'/(sid+'.json'))
        states.update(w['status'] for w in original['words'])
        batch = pad([(arrays, meta)], 'audio')
        assert bool(batch['sample_available'][0]) != source['zero_audio']
        assert not batch['values'][~batch['dimension_mask']].any()
        count = len(original['words']); accepted = original['automatic_eligible_words']
        faces = arrays['face_counts']
        observability.append({'sample_id': sid, 'words': count, 'candidates': accepted,
            'coverage': accepted/count, 'zero_audio': source['zero_audio'],
            'unresolved_reasons': dict(Counter(w['status'] for w in original['words'] if not w['automatic_eligible'])),
            'audio_rows': len(arrays['audio']), 'audio_valid_dimensions': batch['dimension_mask'].sum(axis=(0,1)).tolist(),
            'video_rows': len(arrays['scene']), 'no_face_frames': int((faces==0).sum()),
            'single_face_frames': int((faces==1).sum()), 'multiple_face_frames': int((faces>1).sum()),
            'face_events': len(arrays['faces']), 'crop_edge_events': int(arrays['face_touches_crop_edge'].sum())})
        path = Path(inputs['source_root'])/source['path']
        assert digest(path) == source['sha256']
        pcm = np.frombuffer(run(['ffmpeg','-nostdin','-v','error','-i',str(path),
                               '-map','0:a:0','-f','f32le','-']), '<f4').reshape(-1, source['audio']['channels'])
        wave = torchaudio.functional.resample(torch.from_numpy(pcm.mean(1).copy()),
                    int(source['audio']['sample_rate']), 16000).numpy()
        variants = [('identity', wave, 0.), ('half_gain', wave*.5, 0.),
                    ('leading_silence_1s', np.r_[np.zeros(16000, np.float32), wave], 1.)]
        for name, audio, shift in variants:
            result = infer(engine, audio, meta['words'], source['audio']['start_s'], cfg)
            if name == 'identity':
                assert result['words'] == original['words'], ('identity', sid)
            compared, retained, lost, added = [], 0, 0, 0
            outputs = []
            for before, after in zip(original['words'], result['words']):
                b, a = before['automatic_eligible'], after['automatic_eligible']
                retained += bool(b and a); lost += bool(b and not a); added += bool(a and not b)
                residual = None
                if b and a:
                    residual = max(abs(np.asarray(after['interval_s'])-shift-np.asarray(before['interval_s'])))
                    compared.append(float(residual))
                outputs.append({'word_index':after['word_index'], 'candidate':a,
                                'status':after['status'], 'interval_s':after['interval_s'],
                                'shift_corrected_boundary_residual_s':residual})
            experiments.append({'sample_id':sid, 'variant':name, 'reference_candidates':accepted,
                'candidates':result['automatic_eligible_words'], 'retained':retained,'lost':lost,'added':added,
                'boundary_residual_s':distribution(compared), 'words':outputs})
        write(args.output/'sample_records'/(sid+'.json'),
              {'observability':observability[-1], 'experiments':experiments[-3:]})
        print(index+1, sid, accepted, flush=True)
    engine.face.close()
    # Known analytical signals test descriptor behavior, not emotion prediction.
    tones = []
    time = np.arange(32000)/16000
    for hz in [80, 120, 200, 300, 400]:
        a, mask, _ = acoustic((.2*np.sin(2*np.pi*hz*time)).astype(np.float32), method)
        cents = np.abs(1200*np.log2(a[mask[:,-2],-2]/hz))
        tones.append({'frequency_hz':hz, 'valid_pitch_frames':int(mask[:,-2].sum()),
                      'absolute_pitch_error_cents':distribution(cents)})
    signal = (.2*np.sin(2*np.pi*200*time)).astype(np.float32)
    loud, _, _ = acoustic(signal, method); quiet, _, _ = acoustic(signal*.5, method)
    energy_error = float(np.max(np.abs((quiet[:,13]-loud[:,13])-np.log(.5))))
    aggregates = {}
    for variant in protocol['variants']:
        rows = [r for r in experiments if r['variant']==variant]
        errors = [w['shift_corrected_boundary_residual_s'] for r in rows for w in r['words']
                  if w['shift_corrected_boundary_residual_s'] is not None]
        aggregates[variant] = {k:sum(r[k] for r in rows) for k in ['reference_candidates','candidates','retained','lost','added']}
        aggregates[variant].update(boundary_residual_s=distribution(errors),
            retained_within_20ms=int(sum(e<=.020000001 for e in errors)),
            retained_within_100ms=int(sum(e<=.100000001 for e in errors)))
    write(args.output/'results.json', {'status':'passed', 'protocol_sha256':digest(args.output/'protocol.json'),
        'samples':observability, 'word_status_counts':dict(states),
        'coverage_groups':dict(Counter('zero' if r['candidates']==0 else 'full' if r['candidates']==r['words'] else 'partial' for r in observability)),
        'experiments':experiments, 'transformation_summary':aggregates,
        'acoustic_calibration':{'tones':tones,'half_gain_log_rms_max_error':energy_error,
                              'meaning':'Analytical signal checks, not speech pitch ground truth or sentiment prediction.'},
        'human_records_used':False, 'semantic_accuracy_measured':False})
    print(aggregates, flush=True)


if __name__ == '__main__':
    main()
