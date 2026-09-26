import json,subprocess,time,hashlib,csv,importlib.metadata
from pathlib import Path
import numpy as np,opensmile
P=Path('/project');O=Path('/work/egemaps');O.mkdir(exist_ok=True);(O/'samples').mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
lld=opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,feature_level=opensmile.FeatureLevel.LowLevelDescriptors)
fun=opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,feature_level=opensmile.FeatureLevel.Functionals)
config=dict(feature_set='eGeMAPSv02',LLD_columns=lld.feature_names,functional_columns=fun.feature_names,LLD_dim=25,functional_dim=88,sample_rate=16000,versions={p:importlib.metadata.version(p) for p in ['opensmile','audinterface','audformat','numpy','pandas']},script_sha256=sha(Path(__file__)),role='Optional standardized voice-quality feature branch; no training, no replacement of existing features.')
assert len(lld.feature_names)==25 and len(fun.feature_names)==88
(O/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2));start=time.perf_counter();rows=[]
for p in sorted((P/'results/q1_deep_v6/features_v2/samples').glob('*.json'),key=lambda p:json.loads(p.read_text())['table_row_0based']):
 m=json.loads(p.read_text());src=Path('/data')/m['source_path'];assert sha(src)==m['source_sha256'];t=time.perf_counter()
 wave=np.frombuffer(subprocess.check_output(['ffmpeg','-v','error','-i',str(src),'-map','0:a:0','-ac','1','-ar','16000','-f','f32le','pipe:1']),'<f4').copy()
 low=lld.process_signal(wave,16000);high=fun.process_signal(wave,16000)
 intervals=np.c_[low.index.get_level_values('start').total_seconds(),low.index.get_level_values('end').total_seconds()]+m['audio_start_s'];x=low.to_numpy().astype('float32');h=high.to_numpy().astype('float32')
 assert x.shape[1]==25 and h.shape==(1,88);assert np.isfinite(x).all() and np.isfinite(h).all();assert np.all(np.diff(intervals[:,0])>=0)
 np.savez_compressed(O/'samples'/p.with_suffix('.npz').name,lld=x,intervals=intervals,functionals=h)
 source=dict(sample_id=m['sample_id'],source_path=m['source_path'],source_sha256=m['source_sha256'],feature_sha256=sha(O/'samples'/p.with_suffix('.npz').name));(O/'samples'/p.name).write_text(json.dumps(source,indent=2))
 f0=low['F0semitoneFrom27.5Hz_sma3nz'].to_numpy();voice=f0>0
 rows.append(dict(sample_id=m['sample_id'],lld_rows=len(x),lld_dim=25,functionals_dim=88,finite=True,nonzero_pitch_rows=int(voice.sum()),nonzero_pitch_fraction=float(voice.mean()),seconds=time.perf_counter()-t))
 if len(rows)%20==0:print('EXTRACTED',len(rows),flush=True)
with (O/'全100条标准声学特征.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
res=dict(samples=len(rows),LLD_rows=sum(r['lld_rows'] for r in rows),LLD_dim=25,functional_dim=88,all_finite=True,seconds=time.perf_counter()-start,feature_bytes=sum(p.stat().st_size for p in (O/'samples').glob('*.npz')),pitch_nonzero_in_samples=sum(r['nonzero_pitch_rows']>0 for r in rows),note='Nonzero pitch returned by a feature extractor is not speech verification; use source speech diagnostics. This run establishes extraction and availability, not prediction gain.')
(O/'results.json').write_text(json.dumps(res,indent=2));print(json.dumps(res),flush=True)
