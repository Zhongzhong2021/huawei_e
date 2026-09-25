"""Bounded engineering verification, not a new model-selection experiment."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from q2.data import digest, save_json
from q2v2.checkpoint import atomic_save, equal_record, cpu_snapshot
from q2v2.data import load_fold
from q2v2.engine import load_model, train_run
from run_round4 import compare_tensors


def main(root, old2, old4):
    audit = root/'audit'
    audit.mkdir(exist_ok=True)
    started = audit/'recovery_verification_started.json'
    if started.exists():
        raise FileExistsError('Inspect the original verification session; do not restart blindly')
    if not torch.cuda.is_available():
        raise RuntimeError('This verification requires the original GPU environment')
    original_hash = digest(old4/'final/model.pt')
    protocol = json.loads((old4/'study/protocol.json').read_text())
    save_json(started, {'utc': datetime.now(timezone.utc).isoformat(),
        'purpose': '3 real-model GPU-to-CPU serialization checks and ONE exact replay of the original internal fold0 recipe; no new configuration or model selection',
        'model_before_sha256': original_hash,
        'source_hashes': {str(p.relative_to(root)):digest(p) for folder in ['src','scripts','tests'] for p in (root/folder).rglob('*.py')}})
    model, record = load_model(old4/'final/model.pt', 'cuda')
    expected = cpu_snapshot(record)
    record['state_dict'] = model.state_dict()
    durations = []
    for i in range(3):
        t = time.monotonic()
        atomic_save(record, root/'serialization_probe.pt', event_path=audit/'serialization_events.jsonl')
        restored = torch.load(root/'serialization_probe.pt', map_location='cpu', weights_only=True)
        equal_record(expected, restored)
        durations.append(time.monotonic()-t)
        print(json.dumps({'serialization_pass':i+1,'seconds':durations[-1],'tensors':len(restored['state_dict'])}),flush=True)
    del model, record, expected, restored
    torch.cuda.empty_cache()
    save_json(audit/'serialization_roundtrips.json', {'passes':3,'all_tensors_exact':True,'durations':durations})
    fit,val = load_fold(old2,0,'train'),load_fold(old2,0,'valid')
    config = json.loads((old4/'runs/balanced_inverse_fold0/config.json').read_text())
    source = torch.load(old2/'pretrained/bert_base_uncased.pt', map_location='cpu', weights_only=True)
    result = train_run(root,'control_atomic_fold0',config,fit,val,np.arange(30522),source=source,
                       reproduction=True,training_protocol=protocol)
    comparison = compare_tensors(root/'runs/control_atomic_fold0/best.pt',old4/'runs/balanced_inverse_fold0/best.pt')
    reference = json.loads((old4/'runs/balanced_inverse_fold0/metrics.json').read_text())
    comparison['score_delta'] = result['selection']['score']-reference['selection']['score']
    comparison['reference_model_unchanged'] = digest(old4/'final/model.pt') == original_hash
    comparison['checkpoint_file_hash_may_differ'] = 'CPU storage tags and transactional file serialization change bytes, not fitted tensor values'
    save_json(audit/'recovery_verification_complete.json', comparison)
    if not comparison['all_tensors_exact'] or comparison['score_delta'] != 0 or not comparison['reference_model_unchanged']:
        raise RuntimeError('Engineering control changed fitted results')
    print(json.dumps(comparison),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for name in ['root','round2-root','round4-root']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();main(a.root.resolve(),a.round2_root.resolve(),a.round4_root.resolve())
