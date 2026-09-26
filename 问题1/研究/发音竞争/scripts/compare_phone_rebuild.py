"""Check every posterior value and frame interval after a fresh full inference."""
import os,sys,argparse
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];Q=Path(os.environ.get('Q1_PROJECT',R.parents[1]))
sys.path.insert(0,str(Q/'scripts'))
from common import read,write,digest

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reference',type=Path,required=True);args=parser.parse_args()
    current=Path(os.environ.get('Q1_PHONE_POSTERIORS','/workspace/q1-phone-posteriors'))
    a=read(args.reference/'manifest.json');b=read(current/'manifest.json')
    assert a['complete'] and b['complete'] and a['model']==b['model'] and a['input_sha256']==b['input_sha256']
    old={s['sample_id']:s for s in a['records']};new={s['sample_id']:s for s in b['records']};assert set(old)==set(new)
    for sid in old:
        assert old[sid]['source_sha256']==new[sid]['source_sha256'] and old[sid]['decoded_phones']==new[sid]['decoded_phones']
        for root,item in [(args.reference,old[sid]),(current,new[sid])]:assert digest(root/(sid+'.npz'))==item['posterior_sha256']
        with np.load(args.reference/(sid+'.npz')) as x,np.load(current/(sid+'.npz')) as y:
            assert x.files==y.files
            for key in x.files:assert x[key].dtype==y[key].dtype and np.array_equal(x[key],y[key]),(sid,key)
    write(R/'data/reproduction.json',{'status':'passed','samples':len(old),'arrays_exactly_equal':2*len(old),'source_identity_and_decoded_phones_equal':True,
          'original_manifest_sha256':digest(args.reference/'manifest.json'),'fresh_manifest_sha256':digest(current/'manifest.json'),
          'current_inference_script_sha256':digest(R/'scripts/infer_phones.py'),'scope':'Fresh CPU phoneme inference and frame construction on the same host; no claim of arbitrary hardware equivalence.'})
    print('Equal:',len(old),'samples,',len(old)*2,'arrays')
if __name__=='__main__':main()
