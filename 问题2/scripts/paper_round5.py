"""Generate round-five academic supplement from audited decisions and predictions."""
import argparse
import importlib.util
import json
from pathlib import Path

STEM='E题问题2第五轮延后加权实验与论文补充'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def main(root):
    root=Path(root);out=root/'reports';out.mkdir(exist_ok=True)
    spec=importlib.util.spec_from_file_location('round5_style',Path(__file__).with_name('build_paper.py'))
    style=importlib.util.module_from_spec(spec);spec.loader.exec_module(style);style.STEM=STEM
    summary=read(root/'analysis/summary.json');protocol=read(root/'study/protocol.json');audit=read(root/'audit/poststudy_checks.json')
    development=read(root/'study/development.json');frozen=summary['frozen'];conf=summary['confirmed']
    incomplete=summary.get('incomplete_candidates',[])
    blocks=[];f=lambda x:f'{x:.4f}'
    def p(text):blocks.append({'type':'paragraph','text':text})
    def h(text):blocks.append({'type':'heading','level':1,'text':text})
    def table(caption,header,data,widths):blocks.append({'type':'table','caption':caption,'header':header,'rows':data,'widths':widths})
    def eq(n,latex,mathml):blocks.append({'type':'equation','number':n,'latex':latex,'mathml':'<math xmlns="http://www.w3.org/1998/Math/MathML">'+mathml+'</math>'})
    figures=read(root/'figures/round5_figure_manifest.json')
    def figure(n):
        item=figures[n-1];blocks.append({'type':'figure','path':'../figures/'+item['id']+'.png','caption':item['caption']})
    blocks.append({'type':'title','text':'延后类别加权的对照实验与适用边界'})
    h('摘要')
    if frozen['stable_improvement']:
        p(f"在第四轮立即加权模型的基础上，我们检验训练前期采用普通交叉熵、随后启用类别权重的策略。内部三折比较了两种固定切换时点，方案{conf['selected']['name']}通过内部筛选及三种子确认，被确认为本轮替代模型。结论仍限于预定数据划分、训练预算及已重复使用的官方验证条件。")
    elif incomplete:
        p('本轮保留第四轮立即加权模型及其正式预测。预定研究包含前1轮和前2轮不加权的两个方案，其中前1轮方案完成内部三折，但未通过综合分数与类别性能的全部保护条件；前2轮方案因检查点保存中断未完成，不能形成有效性结论。本材料分别记录已完成的负面对照和未完成的运行状态，不把中断当作方法无效的证据。')
    else:
        p('延后加权未通过本轮预定的全部替换条件，因此保留第四轮立即加权模型及其正式预测。我们在相同视频分组、预训练初始化、缺失场景与训练预算下，比较前1轮或前2轮不加权的两种方案，并同时评价缺失鲁棒性、完整输入和类别性能。该负面结果补充了训练策略的适用边界，不表示延后加权在其他任务上必然无效。')
    p('本文区分内部开发、一次候选确认与最终交付。只有同时满足缺失综合分数和所有性能保护条件的方案才进入官方验证；不为某个候选临时放宽规则，也不依据测试集或附件3选择参数。')
    h('1 研究动机与方法来源')
    p('类别权重改变不同类别错误对参数更新的贡献。立即提高较少类别的损失权重可能改善其识别，也可能改变多数类别的决策边界。第四轮结果已经显示这种取舍，因此本轮不再扩大权重强度搜索，而是只改变权重开始作用的时点。研究问题是：先采用普通分类损失适配预训练表示，再启用同样的类别权重，能否改善正向类别表现并保留中性识别能力。')
    p('Cao等在NeurIPS 2019提出了延后重加权训练策略[1]。本研究借鉴其分阶段思想，但保留原有交叉熵、逆频率权重和线性学习率调度，没有采用LDAM间隔损失，也不声称复现原论文的完整算法。赛题仅含三个类别且类别不均衡程度有限，原文在视觉长尾数据上的发现不能直接作为本任务有效性的证据。')
    h('2 延后加权目标函数')
    p('设当前拟合子集含N条样本、C=3个类别，第c类样本数为n_c。类别权重仍只根据当前拟合样本计数确定，内部三折各自估计，不读取验证、测试或附件3标签。令e为从1开始的训练轮次，d为不加权阶段的轮数。立即加权对照取d=0，两个候选分别取d=1和d=2。')
    r,mi,mo,mn,sub,sup,frac,par=style.row,style.mi,style.mo,style.mn,style.sub,style.sup,style.frac,style.par
    nc=sub(mi('n'),mi('c'));wc=sub(mi('w'),mi('c'));uc=sub(mi('u'),mi('c'))
    eq(1,r'u_c=N/(C n_c),\qquad w_c=u_c/(C^{-1}\sum_{k=0}^{C-1}u_k)',
        r(uc,mo('='),frac(mi('N'),r(mi('C'),nc)),mo(','),wc,mo('='),frac(uc,r(frac(mn(1),mi('C')),style.summ(r(mi('k'),mo('='),mn(0)),r(mi('C'),mo('−'),mn(1))),sub(mi('u'),mi('k'))))))
    eq(2,r'w_c^{(e)}=\begin{cases}1,&1\le e\le d,\\w_c,&e>d.\end{cases}',
        r(style.ss(mi('w'),mi('c'),par(mi('e'))),mo('='),mo('{'),'<mtable><mtr><mtd>'+mn(1)+'</mtd><mtd>'+r(mn(1),mo('≤'),mi('e'),mo('≤'),mi('d'))+'</mtd></mtr><mtr><mtd>'+wc+'</mtd><mtd>'+r(mi('e'),mo('>'),mi('d'))+'</mtd></mtr></mtable>'))
    wy=style.ss(mi('w'),sub(mi('y'),mi('i')),par(mi('e')))
    summ=lambda term:r(sub(mo('∑'),r(mi('i'),mo('∈'),mi('B'))),term)
    eq(3,r'\mathcal L_B^{(e)}=-\frac{\sum_{i\in B}w_{y_i}^{(e)}\log p_{i,y_i}}{\sum_{i\in B}w_{y_i}^{(e)}}+\frac{1}{|B|}\sum_{i\in B}H_1(\hat s_i-s_i)',
        r(style.ss(mi('ℒ'),mi('B'),par(mi('e'))),mo('='),mo('−'),frac(summ(r(wy,mi('log'),sub(mi('p'),mi('i,yᵢ')))),summ(wy)),mo('+'),frac(mn(1),r(mo('|'),mi('B'),mo('|'))),summ(r(sub(mi('H'),mn(1)),par(sub(style.hat(mi('s')),mi('i')),mo('−'),sub(mi('s'),mi('i')))))))
    p('式中B为有效批量，y_i为三分类标签，p为分类概率，s_i和预测强度分别为回归的真实值和模型输出；H_1为阈值1的Huber损失。前d轮各类权重均为1，分类项退化为普通交叉熵；之后使用已固定的逆频率权重。两阶段共享参数和优化器状态，切换时不重启模型、学习率或优化器。')
    p('微批量累积仍使用整个有效批量的同一个权重和作为分类分母，避免类别组成不同的微批量改变有效训练目标。该策略不增加推理参数、输入模态或外部标注数据。部署仍使用同一离线模型接口。')
    h('3 预定比较与核验规则')
    table('表1 比较中固定的条件与唯一改变因素',['项目','设定'],[
        ['参照','第四轮立即逆频率加权，d=0'],['新增配置','前1轮不加权与前2轮不加权'],['模型与数据','12层完整词表编码器；原视频分组三折；各折独立标准化'],
        ['开发训练','种子42，最多6轮，早停耐心3轮，有效批量32'],['其余训练条件','连续缺失概率0.7；原学习率、线性调度与Huber系数'],['检查点候选','仅允许e>d的轮次，早停耐心从加权阶段累计'],['确认规则','内部最佳轮数中位数固定训练长度；只确认一个配置，种子42、2026、3407']], [.24,.76])
    p('选择分数沿用缺失场景Macro-F1与归一化MAE的等权组合。与第四轮相比，至少两折的选择分数须提高，三折均值也须提高；完整Macro-F1下降不超过0.01、MAE增加不超过0.03、Accuracy下降不超过0.005、中性F1下降不超过0.01，正向F1还必须严格提高。所有条件在训练前固定。它们是团队内部选型条件，不是官方比赛评分标准。')
    p('在内部合格方案中，只选择平均分数最高的一项进入官方验证确认。最终替换还要求三个相同种子中至少两组配对、以及三种子均值同时通过全部条件。未完成的实验保留状态，不与完整确认方案同等对待。多轮使用官方验证会带来选择偏差，增加保护条件并不能消除这种偏差。')
    p(f"修改训练器后，首先重跑d=0的第1折。其{audit['control']['tensor_count']}个参数张量与第四轮逐项完全相同，最大绝对差为{audit['control']['maximum_absolute_difference']:.1f}，选择分数差为{audit['control']['score_delta']:.1f}。因此可在本环境中复用第四轮同协议对照；这不是跨硬件逐位复现保证。")
    h('4 内部结果与类别取舍')
    if incomplete:
        p('以下表格与图形仅包含完成三折的候选及其立即加权对照。未完成的'+ '、'.join(x['name'] for x in incomplete)+'不绘制均值、不填补缺失折，也不与完成实验进行性能排序。')
    rows=[('立即加权',summary['baseline'])]+[(e['name'],e['summary']) for e in summary['candidates']]
    table('表2 内部三折均值',['配置','S','Macro-F1','MAE','Accuracy','中性F1','正向F1'],[[label,*[f(m[k]) for k in ['score','clean_f1','clean_mae','accuracy','neutral_f1','positive_f1']]] for label,m in rows],[.16,.14,.14,.14,.14,.14,.14])
    figure(1)
    label_map={'score_improved':'平均S未提高','clean_f1':'完整F1保护未通过','clean_mae':'MAE保护未通过','accuracy':'Accuracy保护未通过','neutral_f1':'中性F1保护未通过','positive_f1_improved':'正向F1未提高','fold_improvements':'不足两折提高S'}
    table('表3 逐项选择判定',['方案','提高S折数','固定轮数','是否合格','未通过条件'],[[e['name'],str(e['fold_improvements'])+'/3',str(e['fixed_epochs']),'是' if e['eligible'] else '否','；'.join(label_map[k] for k in row['failed_conditions'].split(';') if k) or '无'] for e,row in zip(summary['candidates'],summary['decisions'])],[.16,.15,.13,.13,.43])
    for e in summary['candidates']:
        diff=e['guard']['deltas']
        p(f"{e['name']}相对立即加权的平均S差为{diff['score']:+.5f}，正向F1差为{diff['positive_f1']:+.4f}，中性F1差为{diff['neutral_f1']:+.4f}，Accuracy差为{diff['accuracy']:+.4f}。这些结果应与表3的所有条件共同解读，不能仅以某一类别或单折的优势判断方案有效。")
    figure(2)
    p('图中的点对应相同视频分组的三折验证，不是三个独立训练种子的重复实验。各配置的最佳轮次由内部验证选择，因此表2和图1反映同预算下训练方案的开发结果，而非未选择过的泛化精度。图2完整保留不利指标与阈值，不通过统一方向翻转或删去场景来隐藏取舍。')
    h('5 确认结果与最终模型')
    if not conf['records'] and incomplete:
        p('完成三折的候选未通过筛选，另一预定候选未完成。原优化截止时间已过，本轮按中断状态关闭，不修改原协议补算正式结果；没有启动新的官方验证确认或重新评价测试集。最终保留第四轮模型，新增分析与目录不代表新的性能提升。')
        for item in incomplete:
            p(f"方案{item['name']}完成的内部折数为{item['completed_folds']}/3。残缺检查点无法由标准加载器读取，原训练进程已不存在；保存停滞与退出的具体原因尚未确定。原文件、配置与日志保留作故障证据，不据其不完整输出推断方法效果。")
    elif not conf['records']:
        p('没有候选满足内部筛选条件，因此本轮未启动新的官方验证确认，也未重新评价测试集。停止这一研究分支节约了计算，并避免为了得到正面结论而继续消耗同一官方验证集。最终模型与第四轮完全相同，不能把新目录或新增分析误写为新的性能提升。')
    else:
        table('表4 官方验证的同种子确认',['种子','ΔS','ΔMacro-F1','ΔMAE','Δ正向F1','通过'],[[str(x['seed']),*[f"{x['deltas'][k]:+.4f}" for k in ['score','clean_f1','clean_mae','positive_f1']],'是' if x['passed'] else '否'] for x in conf['pairs']],[.12,.17,.19,.17,.20,.15])
        p('本轮官方确认判定为'+('通过，采用新模型。' if frozen['stable_improvement'] else '未通过，继续保留第四轮模型。')+'全部种子及性能条件以study/confirmed.json为准，不只报告最有利的种子。')
    p(f"逐样本独立复算覆盖{audit['verified_prediction_scenarios']}组开发及对照预测；拟合类别计数、逐轮权重切换、视频分组不重叠和选择规则均已核验。正式模型SHA256为{frozen['checkpoint_sha256']}。附件3仍需以final目录的当前冻结模型和统一预处理参数生成，不依据专项预测分布调参。")
    h('6 结论与局限')
    if not frozen['stable_improvement'] and incomplete:
        p('完成三折的前1轮延后方案未提供替换第四轮模型的证据，其MAE小幅降低伴随分类指标下降，应作为有限预算下的负面对照保留。前2轮方案尚无完整结果，不能合并写成两个延后方案均无收益，更不能把保存故障归因于训练策略本身。')
    elif not frozen['stable_improvement']:
        p('本轮没有获得支持替换第四轮模型的证据。论文可据此说明：类别权重的时序安排也是需要验证的选择，并非把加权推迟就能同时改善各类别。应保留立即加权方案的已有收益与代价，将两个延后方案作为同协议负面对照，而不是删除未获收益的实验。')
    else:
        p('本轮在固定保护条件下得到可替换方案，但结论仍须与完整的内部开发与三种子确认一起陈述。延后加权的改进是本任务数据上的实证发现，不能据此主张普遍理论优势。')
    p('本研究预定检验两个切换时点，实际完成范围以运行状态为准，训练预算亦有限。现有证据不能排除其他预算、权重定义或模型结构下存在收益，也不能说明分类错误全部由类别不均衡造成。尚未对新的真实缺失机制、独立数据来源或更多训练种子建立广泛证据；后续研究应保持模型选择与最终评价的边界，并控制在同一数据上反复试验的规模。')
    p('本文件补充训练策略的消融证据，不替代此前的数据定义、掩码处理、编码器与融合结构说明。论文整合时将方法放入求解部分，将表2、表3和配对图放入消融分析；最终模型与附件3来源必须保持一致。')
    h('参考文献')
    ref=protocol['reference_paper']
    p('[1] Cao K, Wei C, Gaidon A, Arechiga N, Ma T. Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss[C]. Advances in Neural Information Processing Systems 32, 2019. '+ref['url'])
    markdown=[]
    for b in blocks:
        if b['type']=='title':markdown+=['# '+b['text'],'']
        elif b['type']=='heading':markdown+=['## '+b['text'],'']
        elif b['type']=='paragraph':markdown+=[b['text'],'']
        elif b['type']=='equation':markdown+=['$$',b['latex']+rf'\tag{{{b["number"]}}}','$$','']
        elif b['type']=='figure':markdown+=[f"![{b['caption']}]({b['path']})",'']
        elif b['type']=='table':markdown+=[b['caption'],'','| '+' | '.join(b['header'])+' |','| '+' | '.join(['---']*len(b['header']))+' |']+['| '+' | '.join(map(str,row))+' |' for row in b['rows']]+['']
    (out/f'{STEM}.md').write_text('\n'.join(markdown),encoding='utf-8')
    (out/'round5_content.json').write_text(json.dumps(blocks,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'round5_equations.tex').write_text('\n\n'.join('\\begin{equation}\n'+b['latex']+rf'\tag{{{b["number"]}}}'+'\n\\end{equation}' for b in blocks if b['type']=='equation'),encoding='utf-8')
    try:import docx
    except ImportError:
        print('Markdown complete; native Word requires bundled authoring runtime.');return
    style.build_docx(root,blocks)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
