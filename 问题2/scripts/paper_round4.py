"""Generate consistent editable academic Word and Markdown from round-four evidence."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import statistics

STEM = 'E题问题2第四轮类别平衡实验与论文补充'


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def fmt(value): return f'{value:.4f}'
def ms(values): return f'{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}'


def main(root):
    root = Path(root).resolve(); out = root / 'reports'; out.mkdir(exist_ok=True)
    spec = importlib.util.spec_from_file_location('paper_style', Path(__file__).with_name('build_paper.py'))
    style = importlib.util.module_from_spec(spec); spec.loader.exec_module(style)
    style.STEM = STEM
    r, mi, mo, mn, sub, sup, frac, par, summ = style.row, style.mi, style.mo, style.mn, style.sub, style.sup, style.frac, style.par, style.summ
    blocks = []; tables = []
    def p(text): blocks.append({'type': 'paragraph', 'text': text})
    def h(text): blocks.append({'type': 'heading', 'level': 1, 'text': text})
    def table(caption, header, data, widths):
        b = {'type': 'table', 'caption': caption, 'header': header, 'rows': data, 'widths': widths}; blocks.append(b); tables.append(b)
    def equation(number, latex, mathml):
        blocks.append({'type': 'equation', 'number': number, 'latex': latex,
                       'mathml': '<math xmlns="http://www.w3.org/1998/Math/MathML">' + mathml + '</math>'})
    manifest = read(root / 'figures/round4_figure_manifest.json')
    def figure(number):
        item = next((x for x in manifest if x['id'].startswith(f'r4_{number:02d}_')), None)
        if item: blocks.append({'type': 'figure', 'path': '../figures/' + item['id'] + '.png', 'caption': item['caption']})
    dev = read(root / 'study/development.json'); conf = read(root / 'study/confirmed.json'); freeze = read(root / 'study/frozen.json')
    audit = read(root / 'audit/postfreeze_complete.json'); diag = read(root / 'analysis/output_consistency_decomposition.json')
    control = read(root / 'audit/control_reproduction.json'); protocol = read(root / 'study/protocol.json')
    intervals = list(csv.DictReader((root / 'analysis/paired_cluster_intervals.csv').open(encoding='utf-8-sig')))
    blocks.append({'type': 'title', 'text': '类别平衡训练与中性情感识别的补充验证'})
    h('摘要')
    p('针对既有多模态模型中中性类别识别偏弱的问题，我们在不改变编码器、融合结构与缺失模拟规则的条件下，比较两档仅由拟合样本类别频率估计的分类权重。研究采用原视频分组内部三折进行开发，并以相同随机种子的官方验证结果进行一次确认。数据划分、类别权重来源和微批量累积方式均保留可复算记录。')
    if freeze['stable_improvement']:
        candidate = [x['metrics'] for x in conf['records']]; reference = [x['reference'] for x in conf['records']]
        p(f"候选{conf['selected']['name']}通过预定确认规则。完整输入Macro-F1由{ms([m['clean']['macro_f1'] for m in reference])}变为{ms([m['clean']['macro_f1'] for m in candidate])}，中性类别F1由{ms([m['clean']['class_f1'][1] for m in reference])}变为{ms([m['clean']['class_f1'][1] for m in candidate])}；上述波动为三种子样本标准差。本文同时报告回归与缺失场景表现，不将单项提升等同于所有指标改善。")
    else:
        p('本轮未获得通过完整选择与确认规则的替代模型，因此继续保留第三轮冻结模型。内部开发结果作为类别平衡策略的探索证据保留，不把局部指标改善表述为正式模型提升。')
    p('官方验证已在多轮研究中重复使用，本文结论仅适用于既定开发与确认协议。配对Bootstrap描述固定模型预测下的不确定性，不校正反复模型选择的偏差。')
    h('1 问题诊断与研究假设')
    p('既有模型在三类任务上整体表现并不均衡。训练集负向、中性、正向样本量分别为967、758、1670，最多与最少类别之比约为2.20。该分布偏斜为检验类别平衡策略提供了动机，但不足以证明中性类别错误完全由样本数量造成；语义边界、表示能力和优化过程同样可能影响结果。')
    ref_clean = read(root / 'evaluation/reference/valid/metrics.json')['clean']; cm = ref_clean['confusion_matrix']
    p(f"第三轮种子42模型在完整官方验证集中，中性类别共{sum(cm[1])}条，识别正确{cm[1][1]}条，另有{cm[1][0]}条被判为负向、{cm[1][2]}条被判为正向。其召回率为{cm[1][1]/sum(cm[1]):.2%}，F1为{ref_clean['class_f1'][1]:.4f}。本轮据此提出可检验假设：在其余训练条件固定时，适度增加较少类别的分类损失权重，可能改善中性类别识别，同时保持整体缺失鲁棒性与强度回归性能。")
    p('类别平衡是已有的经验风险调整方法，不构成新的多模态网络理论。逆频率权重的常用形式可见scikit-learn官方文档[1]，加权交叉熵的归一化定义参照PyTorch文档[2]。平方根强度是本任务预先设定的较温和对照，其有效性必须由实验而非命名证明。')
    h('2 类别权重与优化目标')
    p('设当前拟合子集包含N条样本、C=3个类别，第c类计数为n_c。内部三折分别使用各自拟合子集的计数，官方训练确认只使用官方训练集计数。任何内部验证、官方验证、测试或附件3信息均不进入权重估计。定义权重强度α，并将权重的类别算术均值归一化为1：')
    wc, nc = sub(mi('w'), mi('c')), sub(mi('n'), mi('c'))
    raw = sub(mi('u'), mi('c'))
    equation(1, r'u_c=(N/(C n_c))^\alpha,\quad w_c=u_c/(C^{-1}\sum_{k=0}^{C-1}u_k),\quad\alpha\in\{0.5,1\}',
        r(raw, mo('='), sup(par(frac(mi('N'), r(mi('C'), nc))), mi('α')), mo(','), wc, mo('='), frac(raw, r(frac(mn(1),mi('C')), summ(r(mi('k'),mo('='),mn(0)),r(mi('C'),mo('−'),mn(1))),sub(mi('u'),mi('k')))),mo(','),mi('α'),mo('∈'),mo('{'),mn('0.5'),mo(','),mn(1),mo('}')))
    p('α=0.5对应平方根逆频率，α=1对应逆频率；不加权模型是独立的α=0参照。权重的公共缩放不改变下式的分类损失，因为分子、分母同时缩放。若拟合子集中缺少某一类别，代码明确失败，避免通过无穷权重掩盖划分问题。')
    weight_rows = []
    for entry in dev['entries']:
        for fold in range(3):
            prov = read(root / f"runs/{entry['name']}_fold{fold}/provenance.json")
            weight_rows.append([entry['name'], str(fold+1), *[fmt(x) for x in prov['classification_weights']]])
    table('表1 各内部拟合折估计的类别权重', ['配置','折','负向权重','中性权重','正向权重'], weight_rows, [.30,.08,.206,.207,.207])
    p('对于有效批量B，模型输出分类概率p与连续强度预测。分类部分采用权重和归一化的交叉熵，回归部分保持原Huber损失及系数1不变：')
    sb = lambda term: r(sub(mo('∑'),r(mi('i'),mo('∈'),mi('B'))), term)
    wi = sub(mi('w'), sub(mi('y'),mi('i'))); pi = sub(mi('p'),mi('i,yᵢ'))
    equation(2, r'\mathcal L_B=-\frac{\sum_{i\in B}w_{y_i}\log p_{i,y_i}}{\sum_{i\in B}w_{y_i}}+\frac{1}{|B|}\sum_{i\in B}H_1(\hat s_i-s_i)',
        r(sub(mi('ℒ'),mi('B')),mo('='),mo('−'),frac(sb(r(wi,mi('log'),pi)),sb(wi)),mo('+'),frac(mn(1),r(mo('|'),mi('B'),mo('|'))),sb(r(sub(mi('H'),mn(1)),par(sub(style.hat(mi('s')),mi('i')),mo('−'),sub(mi('s'),mi('i')))))))
    p('其中y_i为类别标签，s_i为真实强度，H_1为阈值1的Huber函数。分类权重仅改变训练目标，不增加推理参数或额外网络分支；模型容量、词表和输入接口与第三轮相同。实际含义是改变训练错误的相对代价，而非自动校准预测概率。')
    p('若显存不足而将B拆为若干微批量B_j，不能先独立求各微批量的加权均值再简单平均，因为各微批量的类别组成可能不同。应使用整个有效批量的同一个权重和，累积每段对总目标的贡献：')
    equation(3, r'\mathcal L_B=\sum_j\left[-\frac{\sum_{i\in B_j}w_{y_i}\log p_{i,y_i}}{\sum_{i\in B}w_{y_i}}+\frac{\sum_{i\in B_j}H_1(\hat s_i-s_i)}{|B|}\right]',
        r(sub(mi('ℒ'),mi('B')),mo('='),sub(mo('∑'),mi('j')),mo('['),mo('−'),frac(r(sub(mo('∑'),r(mi('i'),mo('∈'),sub(mi('B'),mi('j')))),wi,mi('log'),pi),sb(wi)),mo('+'),frac(r(sub(mo('∑'),r(mi('i'),mo('∈'),sub(mi('B'),mi('j')))),sub(mi('H'),mn(1)),par(sub(style.hat(mi('s')),mi('i')),mo('−'),sub(mi('s'),mi('i')))),r(mo('|'),mi('B'),mo('|'))),mo(']')))
    p('代码对不同微批量大小与末尾不足整批的情况进行了梯度等价测试。实际实验仍采用有效批量32；若发生显存不足，事件日志记录微批量变化，不终止其他应用程序。本轮不通过额外标注数据、测试集阈值调整或附件3分布反馈改进指标。')
    h('3 预定比较协议与实现一致性')
    p('内部开发继续使用第三轮按视频分组的三折。每个候选仅改变分类权重强度，随机种子42，最多6轮、早停耐心3轮；标准化、原始预训练权重、完整词表、连续缺失训练概率0.7和AdamW参数保持一致。以九类主要缺失场景上的既有综合分数S选择最佳轮次，再用最佳轮数中位数确定官方训练确认的固定轮数。')
    equation(4, r'S=0.5\,\overline{\mathrm{MacroF1}}_{miss}+0.5(1-\overline{\mathrm{MAE}}_{miss}/6)',
        r(mi('S'),mo('='),mn('0.5'),sub(style.bar(mi('MacroF1')),mi('miss')),mo('+'),mn('0.5'),par(mn(1),mo('−'),frac(sub(style.bar(mi('MAE')),mi('miss')),mn(6)))))
    p('S是团队内部选择准则，不是比赛官方评分公式。候选须至少在两个内部折提高S，同时完整输入平均Macro-F1下降不超过0.01、MAE增加不超过0.03。在满足条件者中只将均值最高的一个候选送入官方验证确认。种子42、2026、3407逐一配对第三轮结果；至少两个配对和总体均值同时改善S并通过上述完整输入保护条件，才替换正式模型。')
    p(f"为确认软件修改没有改变对照条件，先重跑不加权配置的第1折。重跑与第三轮对应模型的{control['tensor_count']}个参数张量逐项完全一致，最大绝对差为{control['maximum_absolute_difference']:.1f}，选择分数差为{control['selection_score_delta']:.1f}。该检查支持在同环境与同配方下复用旧对照；不能推广为跨硬件逐位一致保证。")
    h('4 内部三折结果')
    summary_rows = [('不加权参照', dev['reference_summary'])] + [(entry['name'],entry['summary']) for entry in dev['entries']]
    table('表2 内部三折算术均值', ['配置','完整F1','完整MAE','缺失F1','缺失MAE','S'], [[name,*[fmt(m[k]) for k in ['clean_f1','clean_mae','missing_f1','missing_mae','score']]] for name,m in summary_rows], [.24,.152,.152,.152,.152,.152])
    for entry in dev['entries']:
        d = entry['summary']['score'] - dev['reference_summary']['score']
        p(f"{entry['name']}在{entry['fold_improvements']}/3折提高S，平均差为{d:+.5f}；中性F1均值为{entry['summary']['neutral_f1']:.4f}。其内部保护条件判定为{'通过' if entry['eligible'] else '未通过'}。该比较使用各配置在内部验证上选定的最佳轮次，反映受同一预算约束的整体训练方案差异，不是未经选择的泛化性能估计。")
    figure(1)
    h('5 三种子确认与不确定性')
    blocks[-1]['page_break_before'] = True
    if len(conf['records']) == 3:
        getters = [('完整Accuracy',lambda m:m['clean']['accuracy']),('完整Macro-F1', lambda m:m['clean']['macro_f1']),('完整MAE',lambda m:m['clean']['mae']),('完整Pearson',lambda m:m['clean']['pearson']),('负向类别F1',lambda m:m['clean']['class_f1'][0]),('中性类别F1',lambda m:m['clean']['class_f1'][1]),('正向类别F1',lambda m:m['clean']['class_f1'][2]),('缺失平均Macro-F1',lambda m:m['selection']['mean_missing_macro_f1']),('缺失平均MAE',lambda m:m['selection']['mean_missing_mae']),('内部选择分数S',lambda m:m['selection']['score'])]
        data = []
        for label, get in getters:
            old_values = [get(x['reference']) for x in conf['records']]; new_values = [get(x['metrics']) for x in conf['records']]
            data.append([label,ms(old_values),ms(new_values),f'{statistics.mean(new_values)-statistics.mean(old_values):+.4f}'])
        table('表3 官方验证集三种子确认均值与样本标准差', ['指标','第三轮参照','新增候选','均值差'], data, [.25,.29,.29,.17])
        table('表4 同种子配对的预定保护条件', ['种子','ΔS','Δ完整F1','Δ完整MAE','通过'], [[str(x['seed']),f"{x['deltas']['score']:+.5f}",f"{x['deltas']['clean_f1']:+.4f}",f"{x['deltas']['clean_mae']:+.4f}",'是' if x['passed'] else '否'] for x in conf['pairs']], [.15,.23,.23,.24,.15])
        p(f"三种子确认使用固定{conf['selected']['fixed_epochs']}轮训练，不在官方验证上再次挑选轮次、阈值或候选。最终判定为{'通过，替换正式模型' if freeze['stable_improvement'] else '未通过，保留第三轮模型'}。三种子标准差只刻画本次种子波动；样本量较少，不宜把它表述为充分的随机性保证。")
        figure(2)
    else:
        p(f"本轮完成的官方确认种子数为{len(conf['records'])}，不满足三种子完整确认要求，不替换第三轮模型。内部实验结果与未完成状态均保留。")
    if freeze['stable_improvement']:
        core = [x for x in intervals if x['scenario'] in ['clean','missing_average']]
        table('表5 种子42模型差异的95%视频分组配对Bootstrap区间', ['场景','指标','差值','区间'], [[x['scenario'],x['metric'],f"{float(x['difference']):+.4f}",f"[{float(x['lower']):+.4f}, {float(x['upper']):+.4f}]"] for x in core], [.24,.22,.17,.37])
        p('表5采用相同视频分组抽样同时作用于两个模型和各场景，共2000次重采样。区间跨零的指标不能据此声称方向明确；区间未跨零也只是在固定预测与该划分条件下的证据，不能消除前几轮研究及本轮候选选择造成的偏差。')
    h('6 中性类别与双输出的一致性解释')
    if freeze['stable_improvement']: figure(3)
    if freeze['stable_improvement']:
        trade = read(root / 'analysis/class_tradeoff_protocol.json')['records']
        table('表6 类别权衡的补充分组Bootstrap诊断', ['指标','第三轮','第四轮','差值及95%区间'], [[x['metric'],fmt(x['reference']),fmt(x['final']),f"{x['difference']:+.4f} [{x['lower']:+.4f}, {x['upper']:+.4f}]"] for x in trade], [.25,.15,.15,.45])
        p('类别权重改变了不同类别之间的错误代价，不能将宏平均F1提升表述为全面胜出。种子42的结果显示，中性识别改善伴随正向类别F1和整体Accuracy下降；表3保留其余种子的同类指标，表6同时报告全部类别与Accuracy，避免只展示有利类别。这些区间属于固定模型的次级描述分析，未用于选择，也未作多重比较校正。')
    p('三分类与强度回归是不同损失下的两个输出。若将所有非零回归值直接按符号分为正负，而只有精确零对应中性，则预测为中性的样本几乎都会被计入“类别与回归不一致”。这一严格定义容易把接近零的正常连续输出混入真正的正负方向冲突。因此，应分解统计，而不能把总不一致率直接称为模型自相矛盾率。')
    table('表7 完整验证集的严格符号不一致分解', ['模型','总不一致','中性类且回归非零','非中性正负相反','非中性且回归为零'], [[label,*[str(diag[role][key]) for key in ['exact_sign_mismatch','neutral_class_with_nonzero_regression','opposite_nonzero_polarities','nonneutral_class_with_exact_zero_regression']]] for role,label in [('reference','第三轮'),('final','最终冻结模型')]], [.18,.16,.25,.23,.18])
    p('该分解仅用于解释现有诊断指标，没有据官方验证重新设定回归阈值或改写预测。中性类别的精确率、召回率和F1仍按官方离散标签计算，强度误差仍按原连续标签计算。模型没有承诺两个任务在每条样本上满足确定性的符号约束。')
    if freeze['stable_improvement']:
        figure(4)
        p('图4保留各模态的完整输入参照与全部主要缺失比例。文本缺失增大时分类与回归指标整体恶化，而音视频缺失的响应较小且局部非单调。该现象刻画当前模型和既定验证集的响应，不证明音视频天然无用，也不能把少量缺失后的指标上升解释为移除信息必然有益。阴影只表示三种子训练波动，没有覆盖缺失起点的重复抽样不确定性。')
    h('7 结论与论文整合边界')
    if freeze['stable_improvement']:
        p('本轮证据支持在原有编码与融合结构不变的条件下，采用选定的类别权重训练方案作为当前正式模型。结论应连同三折筛选、三种子保护条件及区间的适用范围一起陈述，不宜抽离其中表现最好的一折或一个种子作为总体成绩。未被选中的权重配置仍保留在消融表中。')
    else:
        p('本轮比较补充了类别权重强度对中性识别和整体多任务性能的实证信息，但尚不足以替换第三轮模型。论文应保留这一负面或未确认结果，并继续将第三轮模型及其原专项预测作为正式输出。')
    p(f"冻结后共复算{audit['evaluation_scenarios_recomputed']}组完整评价指标和{audit['training_scenarios_recomputed']}组开发与确认预测指标，并完成附件3的{audit['attachment3_rows']}条预测、单条与批量一致性、模型重载、CPU/GPU容差和空输入有限性检查。模型与标准化参数的校验值见study/frozen.json，图表源文件摘要见figures/round4_figure_manifest.json。")
    p('整合进总论文时，类别权重定义与微批量归一化可加入目标函数与求解部分；表2至表5用于扩展实验与消融部分，表6和表7用于完善局限与错误分析。应同步更新正文的最终模型名称和专项预测来源，避免把第三轮的中性错误计数与第四轮成绩混排。问题3的音视频原因解释不在本研究结论范围内。')
    h('附录 冻结后评价与复现索引')
    test = {role: read(root / f'evaluation/{role}/test/metrics.json')['clean'] for role in ['reference', 'final']}
    table('表8 冻结后的完整测试集描述性结果', ['模型','Accuracy','Macro-F1','MAE','Pearson'], [[label,*[fmt(test[role][key]) for key in ['accuracy','macro_f1','mae','pearson']]] for role,label in [('reference','第三轮'),('final','第四轮')]], [.24,.19,.19,.19,.19])
    p(f"表8使用{test['final']['n']}条测试样本，仅在本轮冻结后计算，不参与权重强度、训练轮数或种子选择。该划分在此前研究中已经被观察，不是未触碰留出集的无偏评价；不能用表8替代内部开发与官方验证确认的完整说明。")
    reproduction = read(root / 'audit/final_reproduction.json')
    p(f"固定种子42的配方重训得到与冻结模型逐项相同的{reproduction['tensor_count']}个参数张量，最大绝对差为{reproduction['maximum_absolute_difference']:.1f}。该结果是本机同环境复现证据，不承诺跨设备或跨数值库版本逐位一致。完整模型、标准化参数和附件3预测分别保存在final/model.pt、final/scaler.npz与final/attachment3_predictions.csv。")
    p('本文件是第三轮完整问题建模材料的新增补充，而非替换原有数据、掩码、预训练编码和融合结构的定义。第四轮代码和协议独立归档；study记录预设比较及选择，runs保存逐轮日志和逐样本预测，analysis保存独立复算、分组Bootstrap和诊断，figures保存SVG、PDF、600 dpi PNG及来源摘要。论文整合时应以第四轮预测为当前输出，并保留第三轮作为明确命名的对照。')
    h('参考文献')
    p('[1] scikit-learn developers. compute_class_weight, scikit-learn 1.7.2 documentation[EB/OL]. https://scikit-learn.org/1.7/modules/generated/sklearn.utils.class_weight.compute_class_weight.html，访问日期2026-09-25。')
    p('[2] PyTorch contributors. CrossEntropyLoss, PyTorch 2.8 documentation[EB/OL]. https://docs.pytorch.org/docs/2.8/generated/torch.nn.CrossEntropyLoss.html，访问日期2026-09-25。')
    markdown = []
    for b in blocks:
        if b['type'] == 'title': markdown += ['# '+b['text'], '']
        elif b['type'] == 'heading': markdown += ['## '+b['text'], '']
        elif b['type'] == 'paragraph': markdown += [b['text'], '']
        elif b['type'] == 'equation': markdown += ['$$', b['latex'] + rf'\tag{{{b["number"]}}}', '$$', '']
        elif b['type'] == 'figure': markdown += [f"![{b['caption']}]({b['path']})", '']
        elif b['type'] == 'table': markdown += [b['caption'], '', '| '+' | '.join(b['header'])+' |', '| '+' | '.join(['---']*len(b['header']))+' |'] + ['| '+' | '.join(map(str,row))+' |' for row in b['rows']] + ['']
    (out / (STEM+'.md')).write_text('\n'.join(markdown), encoding='utf-8')
    (out / 'round4_content.json').write_text(json.dumps(blocks, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'round4_tables.json').write_text(json.dumps(tables, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'round4_equations.tex').write_text('\n\n'.join('\\begin{equation}\n'+b['latex']+rf'\tag{{{b["number"]}}}'+'\n\\end{equation}' for b in blocks if b['type']=='equation'),encoding='utf-8')
    try:
        import docx
    except ImportError:
        print('Markdown and structured evidence complete; native DOCX requires python-docx and the Office math stylesheet.')
        return
    style.build_docx(root, blocks)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path); main(p.parse_args().root)
