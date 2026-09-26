"""Generate the sixth-round evidence report only from audited study records."""
import argparse
import json
from pathlib import Path


def main(root):
    read = lambda name: json.loads((root/name).read_text(encoding='utf-8'))
    dev, conf, audit = [read(p) for p in ['study/development.json', 'study/confirmed.json', 'audit/poststudy_checks.json']]
    assert audit['status'] == 'passed' and not conf['statistical_conditions_passed']
    lines = ['# 第六轮固定概率融合实验与选择边界', '',
        '本轮两个固定融合方案在内部三折均通过筛选，但选中的90%第四轮与10%第三轮组合未通过官方验证三种子确认。因此不替换第四轮正式模型，不改变附件3预测。本结果属于未获稳定改进的探索证据。', '',
        '## 方法与预定协议', '',
        '研究动机是减轻第四轮类别平衡训练对正向类别的性能代价。对相同样本、相同缺失场景、相同内部折或随机种子的已冻结模型，分别平均分类概率与回归强度：', '',
        '$$p_i^{(\\alpha)}=\\alpha p_i^{(4)}+(1-\\alpha)p_i^{(3)},\\qquad \\hat{s}_i^{(\\alpha)}=\\alpha\\hat{s}_i^{(4)}+(1-\\alpha)\\hat{s}_i^{(3)}.$$', '',
        '其中p是按负向、中性、正向排序的三类概率，预测类别为融合概率的最大值索引；s为[-3,3]强度。预先固定α为0.90和0.75，两种输出使用同一权重，没有拟合融合器、温度、决策阈值或额外参数，也不是平均logit。数值计算沿用FP32。', '',
        '组件使用第三轮完整词表12层多模态模型与第四轮逆频率类别加权模型的既有预测。内部三折严格配对视频分组及拟合样本；官方验证按42、2026、3407配对。组件各自沿用此前已经选择的训练轮数，因此本轮研究的是已选模型预测的组合，不是相同权重以外完全单因素的重新训练消融。', '',
        '选择分数为S=0.5×九类缺失平均Macro-F1+0.5×(1−九类缺失平均MAE/6)。至少两折S提高且均值通过全部条件才有资格进入确认：平均S和正向F1严格提高，完整Macro-F1下降不超过0.01、MAE增加不超过0.03、Accuracy下降不超过0.005、中性F1下降不超过0.01。只确认内部均值S最高的合格方案；至少两个配对种子和三种子均值均通过才允许进一步部署验收。', '',
        '原始[预定协议](../study/protocol.json)、[内部选择](../study/internal_selection.json)与[三种子确认](../study/confirmed.json)均保留。本轮没有启动新训练，不读取测试集或附件3，确认失败后没有再试未选中的75/25组合。', '',
        '## 内部三折结果', '',
        '以下为三个内部视频分组验证折的均值，不是官方验证成绩，也不应和三种子表直接作差。差值均由未舍入数值计算。', '',
        '| 方案 | S | ΔS | 完整Macro-F1 | MAE | 中性F1 | 正向F1 | 提高S折数 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    base = dev['reference_summary']
    lines += [f'| 第四轮参照 | {base["score"]:.6f} | — | {base["clean_f1"]:.4f} | {base["clean_mae"]:.4f} | {base["neutral_f1"]:.4f} | {base["positive_f1"]:.4f} | — |']
    for e in dev['entries']:
        m = e['summary']
        lines += [f'| {e["name"]} | {m["score"]:.6f} | {e["guard"]["deltas"]["score"]:+.6f} | {m["clean_f1"]:.4f} | {m["clean_mae"]:.4f} | {m["neutral_f1"]:.4f} | {m["positive_f1"]:.4f} | {e["fold_improvements"]}/3 |']
    lines += ['', '两个方案均通过内部保护条件。90/10组合的平均S最高，因此只对它开展预定确认。内部提升不等于稳定泛化改善。', '',
              '## 官方验证三种子确认', '',
              '官方验证集每个场景728条样本；以下均为90/10组合减第四轮同种子模型。F1、Accuracy与S越大越好，MAE越小越好。', '',
              '| 种子 | ΔS | Δ完整Macro-F1 | ΔMAE | ΔAccuracy | Δ中性F1 | Δ正向F1 | 全部条件 |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for p in conf['pairs']+[{'seed': '均值', **conf['mean_guard']}]:
        d = p['deltas']
        lines += [f'| {p["seed"]} | {d["score"]:+.6f} | {d["clean_f1"]:+.6f} | {d["clean_mae"]:+.6f} | {d["accuracy"]:+.6f} | {d["neutral_f1"]:+.6f} | {d["positive_f1"]:+.6f} | '+('通过' if p['passed'] else '未通过')+' |']
    lines += ['', '只有种子3407通过全部条件。种子42和2026的S下降，2026还违反中性F1保护条件；均值S也下降。虽然平均完整Macro-F1、Accuracy及正向F1改善，预定的缺失鲁棒性与稳定性条件未获满足，因此不将融合模型作为正式成果。三种子逐项结果全部保留，不仅展示有利指标。', '',
              '## 独立复核与论文表述', '',
              f'审计由逐样本预测独立构造混淆矩阵和指标，复核了{audit["verified_unique_prediction_files"]}个不同预测文件、{audit["verified_blend_scenarios"]}组融合场景、原始预测哈希、样本与标签顺序、拟合样本一致性、内部视频分组不重叠及选择判定。每个融合值与FP32凸组合逐项一致。详细记录见[审计结果](../audit/poststudy_checks.json)。', '',
              '可用于论文的结论是：固定概率融合在内部开发中呈现小幅收益，但该收益未在预定三种子确认中稳定保持。保留第四轮模型体现了既定选择规则的约束作用，而不是认为概率融合在所有情境下均无效。', '',
              '本研究只比较两个预先固定的比例，复用了已经开发和选择的组件模型。官方验证已在前几轮反复使用，本次确认不是独立盲测，也不能消除适应性模型选择偏差。本轮未追加Bootstrap或宣称统计显著性；不根据失败结果放宽条件或扩大比例搜索。', '',
              f'当前正式模型SHA256仍为`{audit["official_model_sha256"]}`。运行结束状态见[terminal.json](../study/terminal.json)。完整逐样本档案留在本地独立实验目录，Git同步代码、论文说明、指标及来源哈希。', '']
    out = root/'reports'
    out.mkdir(exist_ok=True)
    path = out/'E题问题2第六轮固定概率融合实验说明.md'
    if path.exists():
        raise FileExistsError(path)
    path.write_text('\n'.join(lines), encoding='utf-8')
    print(path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    main(p.parse_args().root)
