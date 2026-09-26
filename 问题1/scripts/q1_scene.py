# 本程序及代码是在人工智能工具辅助下完成的。
# 工具：OpenAI Codex；模型：GPT-6 Astra (gpt-6-astra)；开发机构：OpenAI；模型发布日期：2026-09-03。
# AI 来源说明于 2026-09-24 补录，未改变算法；参见 AI辅助使用记录与队内审阅要求.md。
"""Generic appearance features retain non-face visual information; never infer emotions."""
import json
from pathlib import Path
import cv2,numpy as np,torch,torchvision
from torchvision.models import resnet18,ResNet18_Weights
from PIL import Image
from q1_extract import pool,sha

torch.set_num_threads(8);device='cuda' if torch.cuda.is_available() else 'cpu'
weights=ResNet18_Weights.IMAGENET1K_V1;model=resnet18(weights=weights).eval().to(device);model.fc=torch.nn.Identity()
transform=weights.transforms();root=Path('results/q1');max_error=0
for f in sorted((root/'samples').glob('*.npz')):
    with np.load(f,allow_pickle=False) as z:s={k:z[k] for k in z.files}
    mp=f.with_suffix('.json');m=json.loads(mp.read_text(encoding='utf-8'))
    cap=cv2.VideoCapture(str(Path('/data')/m['source_path']));wanted=set(s['vision_frame_index'].tolist());batch=[];features=[]
    def flush():
        if batch:
            with torch.inference_mode():features.append(model(torch.stack(batch).to(device)).cpu().numpy())
            batch.clear()
    for i in range(m['original_video_frames']):
        ok,bgr=cap.read();assert ok
        if i in wanted:
            batch.append(transform(Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))))
            if len(batch)==32:flush()
    flush();cap.release();native=np.concatenate(features);assert native.shape==(len(wanted),512)
    saved=native.astype(np.float16);max_error=max(max_error,float(np.max(np.abs(saved.astype(np.float32)-native))))
    # Pool from exactly the stored values, keeping reproduction independent of float16 conversion.
    candidate,present,coverage=pool(s['word_intervals'],s['vision_cells'],saved.astype(np.float32),np.ones(len(native),bool))
    s['scene_native']=saved;s['scene_candidate']=candidate.astype(np.float16)
    s['scene_mask']=present&s['alignment_valid'];s['scene_presence_mask']=present;s['scene_coverage']=coverage
    s['scene']=s['scene_candidate'].copy();s['scene'][~s['scene_mask']]=0
    s['clip_scene']=pool(np.array([[s['vision_cells'][0,0],s['vision_cells'][-1,1]]]),s['vision_cells'],saved.astype(np.float32),np.ones(len(native),bool))[0][0].astype(np.float16)
    assert all(np.isfinite(x).all() for x in (s['scene_native'],s['scene'],s['clip_scene']))
    np.savez_compressed(f,**s);m['feature_sha256']=sha(f);m['shape']={k:list(v.shape) for k,v in s.items()}
    m['scene_valid_frames']=len(native);m['scene_reliable_words']=int(s['scene_mask'].sum())
    m['visual_representation']='61 face dimensions + 512 generic appearance dimensions, separate masks'
    mp.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
config={'model':'torchvision.models.resnet18','weights':'IMAGENET1K_V1','torchvision':torchvision.__version__,
        'weight_url':weights.url,'weight_sha256':sha('/cache/torch/hub/checkpoints/resnet18-f37072fd.pth'),
        'layer':'global average pooling before fc','dimensions':512,'storage_dtype':'float16',
        'max_observed_native_quantization_absolute_error':max_error,'transform':str(transform),
        'script_sha256':sha(__file__),'note':'Generic image appearance; not emotion or active-speaker prediction.'}
(root/'scene_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8');print(json.dumps(config,indent=2))
