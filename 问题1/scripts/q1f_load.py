# 本程序及代码是在人工智能工具辅助下完成的。
# 工具：OpenAI Codex；模型：GPT-6 Astra (gpt-6-astra)；开发机构：OpenAI；模型发布日期：2026-09-03。
# AI 来源说明于 2026-09-24 补录，未改变算法；参见 AI辅助使用记录与队内审阅要求.md。
"""Explicit expression52 + scene512 visual interface for refinement v1.
This interface belongs only to attachment 1 / Question 1.
"""
from pathlib import Path
import json, numpy as np
from q1n_load import load_sample as load_previous, collate

def load_sample(project_root,stem,verify=True):
    root=Path(project_root);old=load_previous(root,stem)
    path=root/'results/q1_refinement_v1/faces'/f'{stem}.npz'
    if verify:
        import hashlib
        def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
        m=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        assert sha(path)==m['output_sha256']
        for folder,key in [('q1/samples','baseline_npz_sha256'),('q1_next/posterior','posterior_npz_sha256'),('q1_next/enhanced','gate_npz_sha256')]:
            assert sha(root/'results'/folder/f'{stem}.npz')==m[key]
    with np.load(path,allow_pickle=False) as z:new={k:z[k] for k in z.files}
    return dict(text=old['text'],text_mask=old['text_mask'],
        audio_soft=old['audio_soft'],audio_soft_mask=old['audio_soft_mask'],
        vision_soft=new['vision_expression_soft'],vision_soft_mask=new['vision_expression_soft_mask'],
        scene_soft=old['scene_soft'],scene_soft_mask=old['scene_soft_mask'],
        alignment_supported=old['alignment_supported'],
        audio_native=old['audio_native'],audio_time=old['audio_time'],audio_cells=old['audio_cells'],
        vision_native=new['vision_expression_native'],vision_valid=new['vision_expression_valid'],
        vision_time=new['vision_expression_time'],vision_cells=new['vision_expression_cells'],
        vision_frame_index=new['vision_expression_frame_index'],vision_bbox=new['vision_expression_bbox'],
        scene_native=old['scene_native'],recovery_route=new['recovery_route'],
        roi_left_top_side=new['roi_left_top_side'],yunet_score=new['yunet_score'],
        vision_soft_candidate=new['vision_expression_soft_candidate'],
        vision_soft_presence=new['vision_expression_soft_presence'],
        vision_effective_seconds=new['vision_expression_effective_seconds'],
        visual_variant='expression52_roi_v1; scene512; no active-speaker identity')
