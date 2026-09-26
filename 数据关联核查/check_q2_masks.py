"""Exercise the current Q2 input contract on the released placeholders."""
import sys
from pathlib import Path
import numpy as np
from check_attachments import load,sha,write
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'问题2/src'))
from q2.data import convert
rows=[]
for p in sorted(Path('/workspace/data/附件3-模态缺失特征样本/对齐版本').glob('*.pkl')):
 x=load(p)['test'];converted,_audit=convert(x,[p.stem]);indices=np.flatnonzero(x['text_bert'][0,0]==100)
 rows.append({'sample':p.stem,'unk_positions':indices.tolist(),'current_text_observed_at_unk':int(converted['observed'][0,indices,0].sum()),'current_audio_observed_at_unk':int(converted['observed'][0,indices,1].sum()),'current_vision_observed_at_unk':int(converted['observed'][0,indices,2].sum())})
write(Path(__file__).with_name('q2_mask_observations.json'),{'source_sha256':sha(ROOT/'问题2/src/q2/data.py'),'samples':rows,'totals':{k:sum(r[k] for r in rows) for k in ['current_text_observed_at_unk','current_audio_observed_at_unk','current_vision_observed_at_unk']},'weights_or_predictions_changed':False})
