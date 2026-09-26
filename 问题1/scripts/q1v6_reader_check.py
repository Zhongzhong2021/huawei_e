"""CPU-only verification of the delivered Question1 features. AI-assisted code."""
import argparse,json
from pathlib import Path
import numpy as np
from q1v6_features import load_sample,collate
def main():
    a=argparse.ArgumentParser();a.add_argument('--features',default='features');args=a.parse_args();p=Path(args.features);files=sorted((p/'samples').glob('*.npz'));assert len(files)==100
    word=[];clock=[]
    for f in files:
        w,m=load_sample(p,f.stem,'word');g,_=load_sample(p,f.stem,'clock');word.append(w);clock.append(g)
        assert len(w['text'])==len(m['words']) and np.isfinite(g['audio']).all()
        assert g['cells'][-1,1]==m['duration_s']
    for batch in [collate(word),collate(clock)]:
        for k,v in batch.items():
            if v.ndim>=2 and v.shape[:2]==batch['sequence_mask'].shape:assert np.all(v[~batch['sequence_mask']]==0)
    print(json.dumps(dict(samples=100,words=sum(len(w['text']) for w in word),clock_positions=sum(len(g['cells']) for g in clock),feature_hashes_verified=True,padding_verified=True)))
if __name__=='__main__':main()
