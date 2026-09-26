"""One coherent current-model manuscript with traceable historical comparisons."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil

STEM = 'E题问题2统一论文素材与当前模型结果'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(docs, root):
    reports = root/'reports'
    reports.mkdir(exist_ok=True)
    spec = importlib.util.spec_from_file_location('paper_style', Path(__file__).with_name('build_paper.py'))
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    style.STEM = STEM
    paths = {'r3': docs/'round3/reports/paper_content.json', 'r4': docs/'round4/reports/round4_content.json'}
    data = {k: read(p) for k, p in paths.items()}
    current = read(root/'analysis/current_evidence.json')
    study_paths = {'r5': docs/'round5_snapshot/study/development.json', 'r6': docs/'round6/study/confirmed.json'}
    r5 = read(study_paths['r5'])
    r6 = read(study_paths['r6'])
    figures = read(root/'figures/current_figure_manifest.json')
    blocks = []
    origins = []
    asset_sources = {}
    def put(block, origin='new', source_index=None):
        b = copy.deepcopy(block)
        b['_origin'] = origin
        b['_source_index'] = source_index
        blocks.append(b)
        return b
    def p(text):
        return put({'type': 'paragraph', 'text': text})
    def h(text, level=1):
        return put({'type': 'heading', 'level': level, 'text': text})
    def take(origin, index):
        b = put(data[origin][index], origin, index)
        if b['type'] == 'figure':
            original = (paths[origin].parent/b['path']).resolve()
            name = origin+'_'+original.stem
            for ext in ['png', 'svg', 'pdf']:
                source = original.with_suffix('.'+ext)
                target = root/'figures'/f'{name}.{ext}'
                if target.exists():
                    assert digest(target) == digest(source)
                else:
                    shutil.copy2(source, target)
                asset_sources[str(target.relative_to(root))] = {'source': str(source), 'sha256': digest(source)}
            b['path'] = '../figures/'+name+'.png'
        return b
    def take_many(origin, indices):
        return [take(origin, i) for i in indices]
    def table(key, caption, headers, rows, widths=None):
        return put({'type': 'table', 'key': key, 'caption': caption, 'header': headers,
                    'rows': rows, 'widths': widths or [1/len(headers)]*len(headers)})
    def equation(key, latex, xml):
        return put({'type': 'equation', 'key': key, 'latex': latex,
                    'mathml': '<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">'+xml+'</math>'})
    def figure(key):
        f = next(f for f in figures if f['id'] == key)
        return put({'type': 'figure', 'key': key, 'path': '../figures/'+key+'.png', 'caption': f['caption']})
    f = lambda value: f'{value:.4f}'
    modname = {'text': '文本', 'audio': '语音', 'vision': '视觉'}
    put({'type': 'title', 'text': '局部模态缺失下的情感预测与类别平衡'})
    h('摘要')
    take('r3', 2)
    p('在预训练编码与多模态融合的基础上，我们采用仅由拟合样本估计的逆频率分类权重，与Huber回归损失联合求解。按视频分组的内部三折用于开发，三个固定种子用于官方验证确认。当前正式模型为第四轮完整词表12层多模态模型，不采用后续未通过确认的延后加权或预测融合方案。')
    means = {}
    for key, fn in {'f1': lambda r: r['clean']['macro_f1'], 'mae': lambda r: r['clean']['mae'],
                    'neutral': lambda r: r['clean']['class_f1'][1], 'missing': lambda r: r['selection']['mean_missing_macro_f1']}.items():
        import statistics
        values = [fn(r['metrics']) for r in current['confirmation']['records']]
        means[key] = f'{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}'
    p(f'官方验证集上，最终方案的完整Macro-F1为{means["f1"]}，中性F1为{means["neutral"]}，九类缺失场景平均Macro-F1为{means["missing"]}，完整MAE为{means["mae"]}；波动表示三种子样本标准差。相对不加权12层方案，中性与宏平均分类性能改善，但Accuracy与正向F1下降，回归基本持平。')
    p('我们通过固定缺失位置与形状、视频分组配对Bootstrap、分类错误与回归残差分析界定结论。连续缺失增强的既有消融未显示稳定优势，延后加权及固定概率融合也未产生符合预定条件的替代模型。附件3全部30条预测已完成，未知标签准确率不能由覆盖率推断。')
    take('r3', 5)
    take_many('r3', range(6, 41))
    # Core definitions retain the established class c / intensity y notation.
    h('类别平衡目标函数', 2)
    take('r4', 6)
    p('以类别频率调整经验风险是已有方法[6,7]。我们只比较不加权、平方根逆频率与逆频率三个强度，不把类别均衡的动机等同于有效性的证明。设当前拟合子集含N条样本，第c类样本量为n_c，C=3。每个内部折独立计数；官方验证、测试和附件3不参与估计。')
    r, mi, mo, mn, sub, frac, par = style.row, style.mi, style.mo, style.mn, style.sub, style.frac, style.par
    eq = take('r4', 11)
    eq['latex'] = eq['latex'].replace(r'\{0.5,1\}', r'\{0,0.5,1\}')
    eq['mathml'] = eq['mathml'].replace('<mn>0.5</mn>', '<mn>0</mn><mo>,</mo><mn>0.5</mn>')
    p('α=0、0.5、1分别对应不加权、平方根逆频率和逆频率。最终采用α=1，权重从第1轮开始作用。公共缩放在加权交叉熵的分子分母中相互抵消，因此将类别权重的算术均值规范为1不改变目标；若当前拟合子集缺少某类，程序明确报错而不生成无穷权重。')
    ci = sub(mi('c'), mi('i'))
    yi = sub(mi('y'), mi('i'))
    wy = sub(mi('w'), ci)
    summ = lambda term, batch='B': r(sub(mo('∑'), r(mi('i'), mo('∈'), mi(batch))), term)
    loss = r(mi('log'), sub(mi('p'), r(mi('i'), mo(','), ci)))
    hub = r(sub(mi('H'), mn(1)), par(sub(style.hat(mi('y')), mi('i')), mo('−'), yi))
    equation('weighted_loss', r'\mathcal L_B=-\frac{\sum_{i\in B}w_{c_i}\log p_{i,c_i}}{\sum_{i\in B}w_{c_i}}+\frac{1}{|B|}\sum_{i\in B}H_1(\hat y_i-y_i)',
             r(sub(mi('ℒ'), mi('B')), mo('='), mo('−'), frac(summ(r(wy, loss)), summ(wy)), mo('+'), frac(mn(1), r(mo('|'), mi('B'), mo('|'))), summ(hub)))
    take('r3', 43)
    p('c_i始终表示离散类别，y_i始终表示连续强度，B表示有效批量；分类概率p的第二下标选择真实类别。加权交叉熵强调较少类别的分类误差，Huber阈值固定为1，对较大回归残差采用线性增长。两任务系数为1∶1。类别权重不增加推理参数，也不将softmax概率自动校准为正确率。')
    h('连续缺失训练', 2)
    take_many('r3', [45, 46])
    take('r3', 47)
    algorithm = take('r3', 48)
    algorithm['rows'][1][1] = '各折只以拟合样本估计标准化参数和类别频率；按阶段协议固定预训练词表。'
    algorithm['rows'][3][1] = '计算加权交叉熵与Huber损失，按有效批量累积梯度，裁剪后以AdamW更新。'
    parameters = take('r3', 49)
    parameters['rows'][-1][1] = '加权交叉熵∶Huber=1∶1'
    p('AdamW采用解耦权重衰减[3]。有效批量固定为32；显存不足时允许微批量缩至16或8并累计贡献，不能对类别组成不同的微批量加权均值简单平均。对B的互不相交分块B_j，分类分母始终使用完整B的权重和，回归分母始终使用|B|。固定轮数确认阶段在训练结束后一次评价，不逐轮利用官方验证选检查点。')
    equation('accumulation', r'\mathcal L_B=\sum_j\left[-\frac{\sum_{i\in B_j}w_{c_i}\log p_{i,c_i}}{\sum_{i\in B}w_{c_i}}+\frac{\sum_{i\in B_j}H_1(\hat y_i-y_i)}{|B|}\right]',
             r(sub(mi('ℒ'), mi('B')), mo('='), sub(mo('∑'), mi('j')), mo('['), mo('−'), frac(summ(r(wy, loss), 'Bⱼ'), summ(wy)), mo('+'), frac(summ(hub, 'Bⱼ'), r(mo('|'), mi('B'), mo('|'))), mo(']')))
    take_many('r3', range(51, 56))
    p('模型开发按轮次顺序进行。第三轮以第二轮两层蒸馏FP32为参照，第四轮以第三轮不加权12层为参照。候选至少在两个内部折提高S，完整输入三折平均Macro-F1下降不超过0.01、MAE增加不超过0.03；合格者仅选择均值S最高者进入三个同种子确认。至少两组配对及均值均提高S并通过完整输入保护条件才允许替换。第四轮确认长度固定为内部最佳轮数中位数4轮。')
    p('第五、六轮在开始前增加Accuracy、中性F1和正向F1条件，用于限制继续牺牲类别性能；这些新条件不追溯改写第四轮的选择过程。内部交叉验证已用于选择，官方验证也已多轮使用，均不应表述为从未参与开发的独立验证。既有测试结果仅作冻结后的描述，不用来挑选本文方案。')
    h('5 模型开发与主结果')
    h('预训练方案与缺失增强对照', 2)
    take_many('r3', [58, 59, 60, 61, 63, 62])
    p('上述缺失增强消融属于第三轮不加权12层结构。当前加权模型沿用该训练设置，尚无同协议加权模型的关闭增强对照，因此不能将第三轮结果推广为当前模型增强收益的证明。采用增强是预定训练流程的选择，不作为已经证实的独立创新结论。')
    h('类别权重的内部配对比较', 2)
    take_many('r4', [21, 24, 26, 27, 28, 29])
    h('三种子主结果与性能取舍', 2)
    take('r3', 65)['rows'][1][0] = '第三轮不加权12层'
    main_results = take('r4', 31)
    main_results['header'] = [x.replace('新增候选', '第四轮最终模型') for x in main_results['header']]
    take_many('r4', [32, 33, 34])
    p('第二轮到第三轮的改善涉及编码深度、词表及训练方案变化，不能归为单一因素。第四轮保持编码结构并调整分类损失，三种子完整Macro-F1均改善，但Accuracy和正向F1的均值下降。当前取舍由此前固定的目标与保护条件决定，不代表所有应用场景都会优先选择该模型。')
    h('固定模型的不确定性分析', 2)
    take('r3', 69)
    take_many('r4', [35, 36])
    h('6 局部缺失规律')
    h('模态类型与缺失比例', 2)
    take('r4', 44)
    table('current_main', '最终模型种子42的主缺失场景', ['模态', '比例', 'Macro-F1', 'ΔF1', 'MAE', 'ΔMAE'],
          [[modname[x['modality']], str(x['rate'])+'%', f(x['macro_f1']), f"{x['delta_f1']:+.4f}", f(x['mae']), f"{x['delta_mae']:+.4f}"] for x in current['tables']['main_missing']], [.13, .12, .19, .18, .19, .19])
    text50 = next(x for x in current['tables']['main_missing'] if x['modality'] == 'text' and x['rate'] == 50)
    p(f'种子42在文本连续缺失50%时Macro-F1为{text50["macro_f1"]:.4f}，相对完整输入变化为{text50["delta_f1"]:+.4f}；MAE变化为{text50["delta_mae"]:+.4f}。音视频响应较小且存在非单调变化，可能与当前均值表示、文本主导或噪声移除有关，但本实验不能单独识别这些机制。三种子曲线的阴影只刻画训练种子波动，不覆盖缺失起点重复抽样的不确定性。')
    take('r3', 74)
    h('缺失位置与片段形状', 2)
    shape_figure = figure('current_missing_shapes')
    blocks.remove(shape_figure)
    pos = [current['valid']['final'][f'text_50_{shape}']['macro_f1'] for shape in ['front', 'middle', 'back']]
    p('固定文本缺失比例50%时，前部、中部、后部的Macro-F1分别为'+ '、'.join(f(v) for v in pos)+'。该模型在这一固定划分和实现下对后部缺失更敏感；位置以有效特征顺序定义，没有依据将其解释为特定秒数、情绪转折或结尾词的因果作用。补充热图保留全部36类场景和不利结果。')
    shape_table = table('current_shapes', '同等比例下多个短片段相对单长片段', ['模态', '比例', '单段F1', '多段F1', 'ΔF1', 'ΔMAE'],
          [[modname[x['modality']], str(x['rate'])+'%', f(x['single_f1']), f(x['multi_f1']), f"{x['delta_f1']:+.4f}", f"{x['delta_mae']:+.4f}"] for x in current['tables']['shape_comparison']], [.13, .12, .19, .19, .19, .18])
    blocks.remove(shape_table)
    p('下列片段形状比较的差值均为多个短片段减单长片段。当前模型文本缺失10%和30%时，多段F1较高但MAE也略高；50%时多段F1反而较低而MAE略低。不同评价任务的变化并不一致，不能得出多段缺失普遍更容易或更困难的结论。此处使用第四轮预测，不沿用第三轮方向不同的历史结果。')
    rates = [next(x for x in current['actual_missing_rates'] if x['scenario'] == f'text_{rate}_random') for rate in [10, 30, 50]]
    p('名义比例10%、30%、50%对应的样本平均实际位置比例依次为'+ '、'.join(f"{float(x['actual_ratio_mean']):.2%}" for x in rates)+'。差异源于按有效长度取整及至少移除一个位置；若所选位置原已不可观测，新增信息损失少于所选位置数。这两类数量分别保留在实验档案中。')
    blocks.extend([shape_figure, shape_table])
    h('7 分类错误与回归诊断')
    take_many('r4', [38, 39, 40, 41, 42, 43])
    clean = current['valid']['final']['clean']
    cm = clean['confusion_matrix']
    p(f'当前种子42模型中，中性样本{sum(cm[1])}条，正确识别{cm[1][1]}条，误判负向{cm[1][0]}条、正向{cm[1][2]}条。该计数对应第四轮完整输入结果，不能与第三轮混淆矩阵混用。')
    table('current_cases', '按固定规则选取的当前模型验证案例', ['样本标识', '选取规则', '真类→预测', '真实强度', '预测强度'],
          [[x['sample_id'], '错误最大MAE' if x['selection_rule'].startswith('wrong') else '正确中位MAE', f"{x['true_class']}→{x['predicted_class']}", f(x['true_intensity']), f(x['predicted_intensity'])] for x in current['tables']['cases']], [.30, .24, .16, .15, .15])
    take('r3', 88)
    figure('current_regression')
    h('8 后续优化的负面证据')
    p('为减少第四轮的类别性能代价，第五轮预定检验延后启用权重，第六轮预定检验两个固定概率融合比例。两轮均要求相对第四轮平均S和正向F1严格提高，完整Macro-F1下降不超过0.01、MAE增加不超过0.03、Accuracy下降不超过0.005、中性F1下降不超过0.01，并要求至少两折改善S；确认还要求至少两个配对种子与均值均通过。')
    p('延后加权借鉴Cao等的分阶段思想[8]，但保持交叉熵、逆频率权重和原学习率计划，不采用LDAM损失，也不宣称复现其完整算法。前1轮不加权方案完成三折；前2轮方案因检查点保存中断而没有完整折结果，原预算结束后以中断状态关闭，不将故障推断为策略无效。')
    e5 = r5['entries'][0]
    table('defer', '前1轮延后加权的内部结果差值', ['比较', 'ΔS', 'Δ完整F1', 'ΔMAE', 'ΔAccuracy', 'Δ正向F1', '提高S折数'],
          [['延后1轮−立即', *[f"{e5['guard']['deltas'][k]:+.6f}" for k in ['score', 'clean_f1', 'clean_mae', 'accuracy', 'positive_f1']], str(e5['fold_improvements'])+'/3']], [.16, .14, .14, .14, .14, .14, .14])
    p('前1轮延后方案未通过内部筛选，没有开展官方验证确认。其回归误差略低伴随分类表现下降；前2轮未完成方案不进入均值或排名。该分支未产生正式替代模型。')
    p('第六轮将相同折或种子的第三轮与第四轮分类概率、回归输出作同一固定权重的凸组合，第四轮占比预先限定为0.90或0.75。不拟合融合器、温度或阈值，也不重新训练组件。两个比例均通过内部三折筛选，仅内部均值S较高的90/10进入官方验证确认。')
    table('blend', '90比10固定融合相对第四轮的官方验证确认', ['种子', 'ΔS', 'Δ完整F1', 'ΔMAE', 'Δ中性F1', 'Δ正向F1', '全部条件'],
          [[str(x['seed']), *[f"{x['deltas'][k]:+.6f}" for k in ['score', 'clean_f1', 'clean_mae', 'neutral_f1', 'positive_f1']], '通过' if x['passed'] else '未通过'] for x in r6['pairs']+[{'seed':'均值', **r6['mean_guard']}]], [.12, .15, .15, .15, .15, .15, .13])
    p('90/10只有1/3配对种子通过全部条件，平均S也下降，因此不替换第四轮。确认后没有再对未选中的75/25使用官方验证，也没有放宽条件。内部小幅收益未能稳定保持，属于有限搜索下的负面实证；不能推广为所有融合方法无效。两组件和官方验证均已被使用多轮，确认不是独立盲测。')
    h('9 专项预测与模型复杂度')
    take('r3', 90)
    table('current_special', '附件3全部30条当前模型预测', ['文件编号', '预测类别', '情感强度', '最大类概率'],
          [[x['sample_id'].split('::')[0].replace('.pkl',''), ['负向','中性','正向'][int(x['class_id'])], f(float(x['intensity'])), f(max(float(x['prob_'+c]) for c in ['negative','neutral','positive']))] for x in current['tables']['attachment3']], [.34, .20, .23, .23])
    p('专项CSV保留sample_id、scenario、class_id、sentiment、intensity和三类概率，表内数值为显示而舍入，CSV保留原计算精度。稳定标识使用文件名和文件内序号，不能伪装为原始视频ID。附件3没有标签，其预测分布不用于调参，最大类概率也不是已校准的置信度。')
    take_many('r3', [94, 95])
    h('10 结论与适用边界')
    p('本研究建立了与赛题对齐特征接口一致的局部缺失多任务预测流程。预训练模型、加权交叉熵与Huber损失均为已有方法；任务适配主要体现在编码前移除缺失信息、有效位置与观测状态分离、仅拟合训练集预处理统计量，以及固定场景和视频分组下的配对评价。没有提出新的预训练理论或证明最优性。')
    p('正式选择的第四轮模型较不加权参照提高中性识别与宏平均分类性能，同时承担Accuracy和正向F1下降的代价。固定模型Bootstrap的总体F1差区间跨零，因此不能称为所有指标显著提升。缺失形状的影响依赖具体模型、模态、比例与任务；连续缺失增强、延后加权和概率融合的未获收益结果均完整保留。')
    take('r3', 96)
    p('当前正式输出固定来自第四轮种子42模型，而三种子均值用于描述方案稳定性；不能将均值误写为某一部署模型的实际输出指标。附件3全量预测及复现校验已形成记录，真实缺失分布下的效果仍需独立数据评价。问题3要求的音视频原因解释不属于本文数值结果能够证明的范围。')
    h('附录A 蒸馏对照与训练诊断')
    take('r3', 107)
    take('r3', 42)['key'] = 'unweighted_base'
    take('r3', 108)
    p('附录中的基础损失是未加权交叉熵加Huber，不能用正文当前模型的加权损失替代第二轮实际目标。蒸馏温度T_d=2，KL按类别求和后对样本平均；回归蒸馏为教师与学生强度均方差，特征项为768维池化表示经无仿射LayerNorm后的均方差。教师停止梯度。该基准正式训练7轮，与当前固定4轮方案的差异属于完整配方比较。')
    figure('current_learning')
    p('训练损失持续下降而内部选择分数可能波动或回落，因此训练轮数由预定验证规则确定，不以最后一轮训练损失最小作为模型最好证据。微批量梯度等价性、空输入有限性和检查点保存保护均有自动测试；这类工程检查证明实现约束，不直接证明预测能力。')
    h('附录B 冻结后的描述性测试')
    take_many('r4', [51, 52])
    h('附录C 复现与来源')
    take('r3', 116)
    take('r4', 53)
    p('第四轮冻结模型SHA256为'+current['frozen']['checkpoint_sha256']+'。推理加载对应的model.pt与scaler.npz，不依赖在线下载或教师。第五轮原始记录与残缺检查点保留，后续保存器改造及复现作为单独工程证据，不将其写成原故障原因已经查明。')
    p('本文表格与图形由已冻结逐样本预测、开发记录和确认记录生成。第三轮结构定义、历史消融、第四轮当前成绩、第五六轮负面研究在来源清单中分别标识；表号、图号与公式号在本文范围内连续编号。正式参赛前仍需与整队总稿统一格式、摘要和附件限制。')
    h('参考文献')
    take_many('r3', range(101, 106))
    take_many('r4', [56, 57])
    p('[8] Cao K, Wei C, Gaidon A, Arechiga N, Ma T. Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss[C]. Advances in Neural Information Processing Systems 32, 2019. https://papers.neurips.cc/paper_files/paper/2019/hash/621461af90cadfdaf0e8d4cc25129f91-Abstract.html')
    # Assign all numbering once and resolve references by original source namespace.
    mapping = {}
    counts = {'equation': 0, 'figure': 0, 'table': 0, 'algorithm': 0}
    for b in blocks:
        typ = b['type']
        if typ == 'table' and len(b['rows']) <= 10:
            b['keep_together'] = True
        if typ not in ['equation', 'figure', 'table']:
            continue
        original = b.get('caption', '')
        group = 'algorithm' if original.startswith('算法') else typ
        counts[group] += 1
        num = counts[group]
        if typ == 'equation':
            old = b.get('number')
            if old is not None:
                mapping[f"{b['_origin']}:equation:{old}"] = num
            b['number'] = num
        else:
            prefix = '算法' if group == 'algorithm' else ('图' if typ == 'figure' else '表')
            found = re.match(r'^(图|表|算法)([A-Z]?\d+)\s*', original)
            if found:
                mapping[f"{b['_origin']}:{group}:{found[2]}"] = num
            b['caption'] = prefix+str(num)+' '+re.sub(r'^(图|表|算法)[A-Z]?\d+\s*', '', original)
        if b.get('key'):
            mapping[b['key']] = num
    for b in blocks:
        origin = b['_origin']
        if 'text' in b and origin in ['r3', 'r4']:
            text = b['text']
            def ref(m):
                typ = {'式': 'equation', '表': 'table', '图': 'figure', '算法': 'algorithm'}[m[1]]
                key = f'{origin}:{typ}:{m[2]}'
                if key not in mapping:
                    raise ValueError('Unresolved source reference '+key+' at block '+str(b['_source_index']))
                return m[1]+('（'+str(mapping[key])+'）' if typ == 'equation' else str(mapping[key]))
            text = re.sub(r'(?<!词)(式|表|图|算法)[（(]?([A-Z]?\d+)[）)]?', ref, text)
            if origin == 'r4':
                text = re.sub(r'\[([12])\]', lambda m:'['+str(int(m[1])+5)+']', text)
                text = text.replace('本轮', '第四轮')
            elif b['_source_index'] >= 57:
                text = text.replace('本轮', '第三轮')
            b['text'] = text
        origins.append({'block': len(origins), 'type': b['type'], 'origin': origin, 'source_index': b['_source_index']})
    markdown = []
    for b in blocks:
        if b['type'] in ['title', 'heading']:
            markdown += ['#'*(1 if b['type']=='title' else b['level']+1)+' '+b['text'], '']
        elif b['type'] == 'paragraph':
            markdown += [b['text'], '']
        elif b['type'] == 'equation':
            markdown += ['$$', b['latex']+rf'\tag{{{b["number"]}}}', '$$', '']
        elif b['type'] == 'figure':
            markdown += [f"![{b['caption']}]({b['path']})", '']
        elif b['type'] == 'table':
            markdown += [b['caption'], '', '| '+' | '.join(b['header'])+' |', '| '+' | '.join(['---']*len(b['header']))+' |']
            markdown += ['| '+' | '.join(map(str, row))+' |' for row in b['rows']]+['']
    def write(name, value):
        (reports/name).write_text(value, encoding='utf-8')
    write(STEM+'.md', '\n'.join(markdown))
    write('current_content.json', json.dumps(blocks, ensure_ascii=False, indent=2))
    write('current_equations.tex', '\n\n'.join('\\begin{equation}\n'+b['latex']+rf'\tag{{{b["number"]}}}'+'\n\\end{equation}' for b in blocks if b['type']=='equation'))
    provenance = {'generated_utc': datetime.now(timezone.utc).isoformat(), 'current_model_sha256': current['frozen']['checkpoint_sha256'],
        'source_documents': {k: {'path': str(p), 'sha256': digest(p)} for k,p in paths.items()},
        'source_studies': {k: {'path': str(p), 'sha256': digest(p)} for k,p in study_paths.items()},
        'word_builder_sha256': digest(Path(__file__).with_name('build_paper.py')),
        'evidence_sha256': digest(root/'analysis/current_evidence.json'), 'script_sha256': digest(Path(__file__)),
        'copied_figure_sources': asset_sources, 'blocks': origins, 'numbering': mapping, 'counts': counts,
        'note': 'Historical documents and model records were not overwritten; this is an integrated publication, not a new model round.'}
    write('current_provenance.json', json.dumps(provenance, ensure_ascii=False, indent=2))
    style.build_docx(root, blocks)
    print(json.dumps(counts))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--docs', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    main(a.docs.resolve(), a.root.resolve())
