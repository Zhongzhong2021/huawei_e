# 本程序及代码是在人工智能工具辅助下完成的。
# 工具：OpenAI Codex；模型：GPT-6 Astra (gpt-6-astra)；开发机构：OpenAI；模型发布日期：2026-09-03。
# AI 来源说明于 2026-09-24 补录，未改变算法；参见 AI辅助使用记录与队内审阅要求.md。
"""Load baseline, posterior candidates, and conservative evidence masks together."""
from pathlib import Path
import numpy as np

def load_sample(project_root,stem):
    root=Path(project_root)
    def read(p):
        with np.load(p,allow_pickle=False) as f:return {k:f[k] for k in f.files}
    base=read(root/'results/q1/samples'/f'{stem}.npz')
    posterior=read(root/'results/q1_next/posterior'/f'{stem}.npz')
    evidence=read(root/'results/q1_next/enhanced'/f'{stem}.npz')
    result={**base,**posterior,**evidence}
    for name in ('audio','vision','scene'):
        value=posterior[name+'_soft_candidate'].copy();value[~evidence[name+'_soft_mask']]=0
        result[name+'_soft']=value
    return result

def collate(samples):
    lengths=np.array([len(s['text']) for s in samples]);size=int(lengths.max())
    result={'lengths':lengths,'sequence_mask':np.arange(size)[None,:]<lengths[:,None]}
    for name,mask in [('text','text_mask'),('audio_soft','audio_soft_mask'),('vision_soft','vision_soft_mask'),('scene_soft','scene_soft_mask')]:
        x=np.zeros((len(samples),size,samples[0][name].shape[1]),np.float32);m=np.zeros((len(samples),size),bool)
        for i,s in enumerate(samples):x[i,:lengths[i]]=s[name];m[i,:lengths[i]]=s[mask]
        result[name]=x;result[mask]=m
    return result
