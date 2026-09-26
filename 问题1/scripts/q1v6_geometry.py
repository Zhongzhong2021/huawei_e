import torch,torchaudio,json,gc
from pathlib import Path
rows=[]
for name in ['MMS_FA','WAV2VEC2_ASR_BASE_960H']:
    b=getattr(torchaudio.pipelines,name);m=b.get_model(with_star=False) if name=='MMS_FA' else b.get_model();layers=[];jump=1;receptive=1
    for path,c in m.named_modules():
        if isinstance(c,torch.nn.Conv1d) and 'feature_extractor' in path:
            assert c.padding==(0,) and c.dilation==(1,)
            layers.append(dict(path=path,kernel=c.kernel_size[0],stride=c.stride[0]));receptive+=(c.kernel_size[0]-1)*jump;jump*=c.stride[0]
    assert jump==320 and receptive==400
    rows.append(dict(model=name,layers=layers,convolution_stride_samples=jump,convolution_receptive_field_samples=receptive,note='Transformer context extends beyond this convolution window. Centers anchor the feature clock, not true phonetic boundaries.'));del m;gc.collect()
Path('results/q1_deep_v6/conv_geometry_check.json').write_text(json.dumps(rows,indent=2));print(rows)
