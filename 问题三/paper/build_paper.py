"""Regenerate the formal paper, figures and all twenty explanation cards from frozen results."""
from pathlib import Path
import csv, json, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import font_manager
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from lxml import etree
from latex2mathml.converter import convert as latex_to_mathml

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT/'paper'
FIG = PAPER/'figures'
FIG.mkdir(exist_ok=True)

def read(p): return json.loads((ROOT/p).read_text())
def rows(p):
    with (ROOT/p).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def records(p): return [json.loads(x) for x in (ROOT/p).read_text().splitlines() if x]
def save(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=200,bbox_inches='tight')
    svg=FIG/(name+'.svg');fig.savefig(svg,bbox_inches='tight')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)

MET = read('results/uncertainty/metrics.json')
CON = read('results/contributions/contribution_summary.json')['summary']
CMP = rows('results/model_comparison.csv')
SENS = read('results/sensitivity/controlled_comparisons.json')
SWEEP = rows('results/sensitivity/uncertainty_all_candidates.csv')
BASE = read('results/baseline_parameters/comparisons.json')['records']
assert [r['code'] for r in BASE]==['S0','S1','S2','S3','S4','S5','T0','T1']
assert all(r['config']['train']['seed']==17 and r['config']['train']['max_epochs']==10 for r in BASE)
for code,family in [('S5','selfmm'),('T1','tetfn_source')]:
    ref=next(r for r in CMP if r['family']==family)
    item=next(r for r in BASE if r['code']==code)
    for split in ['valid','test']:
        for metric in ['accuracy','macro_f1','mae']:
            assert abs(item[split][metric]-float(ref[split+'_'+metric]))<1e-7
assert len(SWEEP)==22
assert {(r['ordinal_head'],r['uncertainty_fusion']) for r in SENS['structures']}=={(False,False),(False,True),(True,False),(True,True)}
assert SENS['structures'][-1]['valid']==MET['valid']['final']
for group in 'AB':
    assert {r['multiplier'] for r in SENS['learning_rates'] if r['group']==group}=={.5,1,2}
MEDIA = {x['id']:x for x in records('results/media_mapping.jsonl')}
EXP = {x['id']:x for x in records('results/uncertainty/special_explanations.jsonl')}
PRED = {x['id']:x for x in rows('results/uncertainty/special_predictions.csv')}
SHARE = {x['id']:x for x in rows('results/contributions/attachment4_contributions.csv')}
def strength(v):
    v=float(v)
    return f'{v:+.1e}' if 0<abs(v)<1e-4 else f'{v:+.4f}'

CN = ['负向','中性','正向']
COLORS = ['#277DA8','#D89635','#5B9982']
font_manager.fontManager.addfont(str(PAPER/'assets/Q3PaperSans.otf'))
plt.rcParams.update({'font.family':['Q3 Paper Sans','DejaVu Sans'],'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'path'})

# A compact workflow distinguishes parameter learning, selection and inference.
fig,ax=plt.subplots(figsize=(11,4.1));ax.set(xlim=(0,11),ylim=(0,4.1));ax.axis('off')
steps=[(.1,2.65,3.1,'附件二训练集\n3395 条：统计估计与参数学习'),
       (3.95,2.65,3.1,'附件二验证集\n728 条：检查点与策略选择'),
       (7.8,2.65,3.0,'固定模型与决策规则\n不确定性融合＋局部证据读出'),
       (4.15,.45,2.9,'附件二测试集（727 条）\n基础性能、误差与贡献分析'),
       (7.9,.45,2.9,'附件四（20 条）\n全量预测、解释卡与证据定位')]
for x,y,w,t in steps:
    ax.add_patch(FancyBboxPatch((x,y),w,1.05,boxstyle='round,pad=0.07',fc='#EDF3F7',ec='#597080'))
    ax.text(x+w/2,y+.525,t,ha='center',va='center',fontsize=10)
for a,b in [((3.2,3.175),(3.88,3.175)),((7.05,3.175),(7.73,3.175)),((9.3,2.58),(9.3,1.58))]:
    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#597080',lw=1.4))
ax.plot([9.3,5.6,5.6],[2.1,2.1,1.58],color='#597080',lw=1.4)
ax.annotate('',xy=(5.6,1.5),xytext=(5.6,1.75),arrowprops=dict(arrowstyle='->',color='#597080',lw=1.4))
ax.text(.15,1.15,'解释链条\n预测目标 → 模态净贡献 → 局部片段\n保留方向、原文索引与估计时间',va='center',fontsize=10,color='#244052')
save(fig,'00_workflow')

# Network: boxes label operations, arrows follow the actual dependency graph.
fig, ax=plt.subplots(figsize=(12,7.2));ax.set_xlim(0,12);ax.set_ylim(-.18,8);ax.axis('off')
def box(x,y,w,h,t,c='#EAF2F7'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.09',fc=c,ec='#597080',lw=1))
    ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=10)
def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#597080',lw=1.2))
for i,(name,enc) in enumerate([('文本 T：词元编号与掩码','BERT（12 层）\n上下文化表示：768 维'),('语音 A：对齐声学特征','训练集统计标准化\n声学表示：74 维'),('视觉 V：对齐视觉特征','训练集统计标准化\n视觉表示：35 维')]):
    x=.1+4*i;box(x,7,3.7,.65,name);box(x,5.75,3.7,.85,enc);arrow((x+1.85,7),(x+1.85,6.6));box(x,4.45,3.7,.85,'局部编码：输入维度 → 64 → 64\n零参考差分 E(x) − E(0)，窗口宽度 1');arrow((x+1.85,5.75),(x+1.85,5.3))
box(.15,2.8,4.0,1,'三个单模态分支\n64 → 4；掩码均值汇聚')
box(4.55,2.8,3.45,1,'TA / TV / AV 成对分支\n秩 8 双线性映射 → 4')
box(8.4,2.8,3.45,1,'单模态净分数＋局部绝对活性\n每模态 8 → 1 对数方差')
for x in [1.95,5.95,9.95]:ax.plot([x,x],[4.36,4.18],color='#597080',lw=1.2)
ax.plot([1.95,9.95],[4.18,4.18],color='#597080',lw=1.2)
for x in [2.15,6.3]:arrow((x,4.18),(x,3.89))
ax.plot([4.15,4.32,4.32,10.1],[3.3,3.3,4.03,4.03],color='#597080',lw=1.2,ls='--');arrow((10.1,4.03),(10.1,3.8));box(8.4,1.35,3.45,1,'归一化精度 p\n单项缩放 κ = Kp\n成对缩放取两端 κ 的几何均值');arrow((10.1,2.8),(10.1,2.35))
box(.15,1.35,7.6,1,'加权单项＋加权交互＋偏置\n回归预激活 q 与三个类别 logit');arrow((2.15,2.8),(2.15,2.35));arrow((6.3,2.8),(6.3,2.35));arrow((8.4,1.85),(7.75,1.85))
box(.15,0,5.5,.85,'联合预测：中性 logit 校准后取最大类别\n3 tanh(q) 经类别一致性处理得到强度','#E8F2ED')
box(6.05,0,5.8,.85,'原生解释：预测类相对竞争类的间隔\n单项证据＋成对项等分核算','#FFF2DE')
arrow((3.1,1.35),(2.9,.85));arrow((5.2,1.35),(8.95,.85));save(fig,'01_network')

hist=rows('results/uncertainty/history.csv');ep=[int(x['epoch']) for x in hist]
fig,axes=plt.subplots(1,2,figsize=(10,3.5))
axes[0].plot(ep,[float(x['train_loss']) for x in hist],'o-',color=COLORS[0]);axes[0].set(xlabel='训练轮次',ylabel='联合训练损失')
axes[1].plot(ep,[float(x['valid_raw_mae']) for x in hist],'o-',color=COLORS[1]);axes[1].set(xlabel='训练轮次',ylabel='验证集原始强度 MAE')
for ax in axes:ax.axvline(5,ls='--',color='#555',label='选中第 5 轮');ax.legend();ax.grid(alpha=.2)
fig.tight_layout();save(fig,'02_training')

fig,axes=plt.subplots(1,2,figsize=(9,3.8))
for ax,split in zip(axes,['valid','test']):
    a=np.asarray(MET[split]['final']['confusion_matrix']);ax.imshow(a,cmap='Blues',vmin=0,vmax=300)
    for (i,j),v in np.ndenumerate(a):ax.text(j,i,str(v),ha='center',va='center',color='white' if v>160 else '#172D3D')
    ax.set(xticks=range(3),yticks=range(3),xticklabels=CN,yticklabels=CN,xlabel='预测类别',ylabel='真实类别',title=f'{"验证集" if split=="valid" else "附件二测试集"}（{a.sum()} 条）')
fig.tight_layout();save(fig,'03_confusion')

valid=rows('results/uncertainty/valid_predictions.csv');y=np.array([float(x['true_intensity']) for x in valid]);yp=np.array([float(x['intensity']) for x in valid])
fig,axes=plt.subplots(1,2,figsize=(9,3.7));axes[0].scatter(y,yp,s=11,alpha=.35,color=COLORS[0]);axes[0].plot([-3,3],[-3,3],'--',color='#777');axes[0].set(xlabel='真实强度',ylabel='最终预测强度',xlim=(-3.1,3.1),ylim=(-3.1,3.1));axes[1].hist(yp-y,bins=25,color=COLORS[0],edgecolor='white');axes[1].axvline(0,color='#555',ls='--');axes[1].set(xlabel='预测强度 − 真实强度',ylabel='样本数');fig.tight_layout();save(fig,'04_regression')

fig,axes=plt.subplots(1,3,figsize=(11,3.8))
for ax,split in zip(axes,['valid','test','special']):
    data=np.array([[CON[split]['classification']['mean_shares'][m] for m in 'TAV'],[CON[split]['regression']['mean_shares'][m] for m in 'TAV'],[CON[split]['mean_precision_weights'][m] for m in 'TAV']])*100
    bottom=np.zeros(3)
    for j,m in enumerate('TAV'):
        ax.bar(range(3),data[:,j],bottom=bottom,color=COLORS[j],label={'T':'文本','A':'语音','V':'视觉'}[m])
        for k,v in enumerate(data[:,j]):
            if v>5:ax.text(k,bottom[k]+v/2,f'{v:.1f}',ha='center',va='center',fontsize=8,color='white' if j==0 else '#172D3D')
        bottom+=data[:,j]
    ax.set(xticks=range(3),xticklabels=['分类间隔','回归 q','可靠性'],ylim=(0,106),ylabel='逐样本占比的均值（%）',title={'valid':'验证集','test':'附件二测试集','special':'附件四'}[split]);ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,1.2))
fig.tight_layout();save(fig,'05_contributions')

# Include the complete seed-17 sweep, alongside complete paired LR groups.
fig,axes=plt.subplots(1,2,figsize=(10,3.8))
axes[0].scatter([float(r['macro_f1']) for r in SWEEP],[float(r['mae']) for r in SWEEP],s=30,color='#9DA9AF',alpha=.8,label='全部22组配置')
for group,color in [('A',COLORS[0]),('B',COLORS[1])]:
    data=[r for r in SENS['learning_rates'] if r['group']==group]
    axes[0].scatter([r['valid']['macro_f1'] for r in data],[r['valid']['mae'] for r in data],s=48,color=color,label='参数组'+group)
    axes[1].plot(range(3),[r['valid']['macro_f1'] for r in data],'o-',color=color,label='参数组'+group)
    for i,r in enumerate(data):axes[1].annotate(f"{r['valid']['macro_f1']:.4f}",(i,r['valid']['macro_f1']),xytext=(0,7 if group=='B' else -15),textcoords='offset points',ha='center',fontsize=9,color=color)
axes[0].set(xlabel='验证集 Macro-F1（越高越好）',ylabel='验证集 MAE（越低越好）')
axes[1].set(xticks=range(3),xticklabels=['0.5倍','1倍','2倍'],xlabel='两组学习率同时缩放的倍率',ylabel='验证集 Macro-F1',ylim=(.597,.645))
for ax in axes:ax.legend(fontsize=9);ax.grid(alpha=.18)
axes[0].legend(loc='upper center',bbox_to_anchor=(.5,1.17),ncol=3,fontsize=9)
fig.tight_layout();save(fig,'07_parameter_comparison')

# Each displayed word is an exact source token mapping; signs retain opposition.
for sid in ['02','14','16']:
    e=EXP[sid];mapping={j:z for z in MEDIA[sid]['entries'] for j in z['source_indices']}
    indices=[(i,s[0]) for i,s in enumerate(e['source_index']) if s[0]>=0]
    vals=[e['local'][i][0] for i,j in indices]
    labels=[f'{j}: {mapping[j]["text"]}' if j in mapping else str(j) for i,j in indices]
    fig,ax=plt.subplots(figsize=(10,max(3,len(vals)*.18)))
    ax.barh(range(len(vals)),vals,color=[COLORS[0] if v>=0 else '#BC6373' for v in vals]);ax.set(yticks=range(len(vals)),yticklabels=labels,xlabel='对预测类别间隔的局部有符号贡献',title=f'附件四样本 {sid}：文本模态局部读出');ax.invert_yaxis();ax.axvline(0,color='#555',lw=.8);ax.tick_params(axis='y',labelsize=11);fig.tight_layout();save(fig,'06_local_'+sid)

# Paper blocks are the single source for Word, Markdown and numbered captions.
blocks=[]
def add(kind,value): blocks.append((kind,value))
def para(s): add('p',s)
def head(s): add('h',s)
def subhead(s): add('h2',s)
def table(headers,data,caption): add('table',(headers,data,caption))
def figure(name,caption): add('figure',(name,re.sub(r'^图\d+\s*','',caption)))
def eq(s): add('eq',s)
add('title','基于不确定性融合与局部可加证据的多模态情感识别')
add('subtitle','问题三：可解释性模型的建立、求解与结果分析')
add('abstract_title','摘　要')
para('针对复杂场景下文本、语音与视觉信息不一致、部分观测缺失以及预测依据难以定位的问题，建立兼顾情感极性分类、强度估计与证据解释的多模态联合模型。本文采用题目给定的对齐版数据，在统一读出结构中保留单模态局部证据与显式成对交互，使预测分数能够按模态及输入位置核算。')
para('在模型建立方面，以BERT提取上下文化文本表示，对语音和视觉特征进行训练集统计标准化；构建零参考局部编码和低秩双线性交互，以单模态回归残差方差估计可靠性，并对实际参与预测的证据进行精度缩放。以Huber损失、分类交叉熵及辅助监督联合优化，利用验证集选择检查点和中性判别策略。解释以预测类别相对最强竞争类别的间隔为目标，通过交互项等分给出三模态净贡献、主要参考模态及关键片段。')
para('在结果检验方面，附件二验证集准确率为65.11%、Macro-F1为0.6323、强度MAE为0.5397；测试集相应为68.64%、0.6493和0.6006。与SELF-MM和TETFN的比赛适配基线比较，主模型在测试分类指标与原生解释能力之间取得较好的折中，但并非所有指标均占优。补充原生结构对照和学习率成组试验表明，组合模块和增大学习率未必改善验证结果。验证集主要错误集中在中性与正向之间，说明弱情感边界仍是模型改进重点。')
para('对附件四20条无标签样本给出全量预测与解释汇总，并结合典型解释卡和局部重要性图展示证据方向与位置。附件四中，文本在分类间隔中的平均净贡献占89.78%，在回归预激活中的占比为73.76%；可靠性权重与贡献比例存在明显差异。结果说明极性判断和强度估计对三模态的利用方式不同，可加读出为这种差异提供了直接的数值解释。')
add('keywords','关键词：多模态情感识别；不确定性融合；局部可加证据；成对交互；可解释性')
add('pagebreak',None)
head('1 问题重述与分析')
subhead('1.1 问题三的任务')
para('问题三要求模型同时给出情感极性、连续强度、三模态作用程度、主要参考模态和关键证据定位。输入由文本、语音、视觉三种模态构成，模型既要完成预测，也要说明该判断由哪些输入片段支撑。本文为问题三独立章节稿，使用附件二给定划分建立和检验模型，对附件四无标签专项样本进行最终预测与解释。')
subhead('1.2 建模难点与总体思路')
para('第一，三模态的数值尺度、观测质量和情感线索不同，简单拼接难以说明各模态作用。第二，模态之间存在补充、冲突与抵消，单独解释各模态不能覆盖融合后的判断。第三，分类与回归具有不同输出空间；用于选择模态的可靠性系数，也不等于模态对某个预测目标的贡献。第四，附件四需要把关键证据关联到原文本和媒体，而对齐特征并未提供精确的音视特征时间戳。')
para('据此，采用“局部编码—单项与成对读出—可靠性缩放—联合预测与核算”的建模路线。以可加结构保证分数可分解，以成对项表达跨模态共同作用，以残差方差抑制相对不可靠的读出；分别解释分类间隔和回归预激活，并通过原词元位置及机器估计时间建立证据索引。总体流程见图1。')
figure('00_workflow','问题三总体技术路线。模型学习与策略选择分别使用训练集和验证集；附件四输出预测与解释。')
head('2 模型假设与符号说明')
subhead('2.1 基本假设')
para('假设1：题目给定的对齐位置可以作为三模态局部读出的共同索引。该索引用于模型计算；机器估计的媒体时间只用于证据回查，不假设其等于原音视特征的精确提取时间。')
para('假设2：无有效数值的音视行不提供局部观测信息，对应项和关联交互置零。缺失状态本身不被赋予正向、负向或中性含义；是否缺失以原始观测掩码判断。')
para('假设3：在上下文化局部编码之后，用单模态项与两两交互近似融合关系，不另设不可分解的融合残差。三方高阶作用未被显式建模，由此得到可核算性与表达能力之间的折中。')
para('假设4：单模态回归残差的条件方差可作为该模态读出的相对可靠性线索。此假设用于构造缩放系数；是否提高泛化性能需由数据检验，方差不直接视为分类置信度或人类证据可信程度。')
subhead('2.2 主要符号')
table(['符号','含义'],[
['m，n；j','模态索引T/A/V；有效局部窗口索引'],
['xₘⱼ，oₘⱼ','局部输入及其有效观测掩码'],
['y，c；ŷ，ĉ','真实强度及类别；预测强度及类别'],
['dₘⱼ；aₘⱼ，iₘₙⱼ','零参考局部表示；单项与成对四维读出'],
['vₘ，pₘ，κₘ','对数方差、归一化精度及单项缩放系数'],
['q，ℓ；β','回归预激活、三分类logit；共享读出偏置'],
['M；Φₘ，ψₘⱼ','预测类别间隔；模态净分数与局部净分数'],
['ρₘ','模态净分数绝对值占比；不包含偏置']], '主要符号及含义')
head('3 数据观察与预处理')
para('标签强度y∈[−3,3]，y<0、y=0、y>0分别对应负向、中性、正向，类别编码为0、1、2。不将接近零但非零的标签重定义为中性。附件二各集合的样本量与类别分布见表2；附件四有20条无标签专项样本，仅列样本量和推理用途。')
table(['集合','样本数','负向','中性','正向','用途'],[['训练集',3395,967,758,1670,'参数学习、统计估计'],['验证集',728,206,184,338,'结构、参数及策略选择'],['附件二测试集',727,207,158,362,'固定模型的基础评价'],['附件四',20,'—','—','—','最终预测与解释']], '题目数据划分与用途')
para('每条样本最多50个对齐位置，给定文本特征768维、语音特征74维、视觉特征35维，另有BERT词元编号、注意力掩码及类型编号。本模型选择text_bert输入预训练BERT[1]，不把text与text_bert视为两个模态。剔除特殊词元和填充位置后保留原序列位置source_index，文本隐状态经LayerNorm后回填到局部窗口。')
para('语音、视觉全零行按无有效数值观测处理。对模态m的第d维，仅在训练集有效观测上计算均值μₘd和总体标准差sₘd，并标准化为(xₘⱼd−μₘd)/sₘd；标准差小于10⁻⁶时置为1。缺失行保持为零且保留掩码，验证、测试和附件四复用训练统计，避免把待评价集合的信息用于归一化参数估计。')
para('模型评估基于既有探索实验：历史研究中曾多次查看附件二测试表现，最终方案同时考虑已有性能与解释要求。因而本文报告的是单种子探索性比较，不能作为完全独立、确认性的模型优越性结论；附件四在正式展示中只报告模型输出。')
head('4 模型建立')
subhead('4.1 上下文化局部证据与成对交互')
para('借鉴神经可加模型[2]与成对交互建模[3]，采用“局部非线性编码—显式加性读出”的结构。BERT提供语言上下文，局部编码保留证据位置，成对分支表达同位置的跨模态共同作用。可靠性缩放在样本层调整这些读出项，不额外设置无法核算的融合残差。预测及解释共享同一次前向中的实际数值。')
figure('01_network','图1 不确定性融合网络。T/A/V为文本/语音/视觉；每个分支的4维输出对应q及负向、中性、正向logit。')
para('记m∈{T,A,V}，j为有效内容窗口。当前窗口宽度为1；每模态局部编码器由逐行线性映射与GELU、窗口拼接线性映射与GELU构成，投影维度与局部隐层均为64。文本BERT隐状态先作LayerNorm，再按source_index回填局部位置。用相同观测结构下的零输入编码作为参考，得到下式。参考为特征坐标基准，不代表真实中性情绪。')
eq('dₘⱼ = Eₘ(xₘⱼ, oₘⱼ) − Eₘ(0, oₘⱼ)')
para('单模态读出为无偏置64→4线性映射。均值汇聚权重ωₘⱼ正比于该窗口内有效观测数；成对权重ωₘₙⱼ以两端权重几何均值构造，限制在共同有效窗口，再按样本归一化。TA、TV、AV各使用独立的秩8双线性投影，分别输出4维交互。')
eq('aₘⱼ = ωₘⱼ Wₘdₘⱼ；  iₘₙⱼ = ωₘₙⱼ Vₘₙ[(Pₘₙdₘⱼ) ⊙ (Qₘₙdₙⱼ)]')
para('P、Q、V均不含偏置；某一模态没有有效观测时，该单项与关联交互严格为零。BERT使单个文本读出含有全句上下文，因此不能把一个局部值解释为只由该词自身产生的独立效应。成对模块连接对齐位置，文本上下文来自BERT，模型不另设音视跨位置注意力或三方高阶交互。')
subhead('4.2 由残差方差驱动的可靠性融合')
para('首先从未进行精度缩放的单项计算每模态四维净分数uₘ=β+Σⱼaₘⱼ，同时统计各输出维度局部读出的绝对活性Σⱼ|aₘⱼ|。拼接后得到8维输入，通过每模态一个8→1线性层预测有界对数方差。训练时活性来自模态丢弃后的单项，单模态监督仍使用丢弃前的实际单项。')
eq('vₘ = −1 + 4 tanh(wₘᵀ[uₘ; Σⱼ|aₘⱼ|] + bₘ)； σₘ² = exp(vₘ)')
eq('pₘ = 1ₘ exp(−vₘ) / Σₙ 1ₙ exp(−vₙ)； κₘ = K pₘ，K = Σₘ 1ₘ')
para('1ₘ表示当前保留且可观测的模态。单项按κₘ缩放，成对项按√(κₘκₙ)缩放。精度相同时κₘ=1，恢复基础加性读出；方差被限制在exp(−5)至exp(3)之间。方差监督针对单模态回归残差，它既不是分类概率校准，也不是融合后预测区间。归一化精度pₘ是可靠性系数，不能直接作为模态贡献百分比。')
eq('s = β + Σₘⱼ κₘaₘⱼ + Σ₍ₘ,ₙ₎∈P,ⱼ √(κₘκₙ)iₘₙⱼ； q=s₀； ℓ=(s₁,s₂,s₃)')
para('P={(T,A),(T,V),(A,V)}为无序模态对的集合，每个成对项只计入一次。')
subhead('4.3 联合输出与模态贡献定义')
para('回归原始输出r=3 tanh(q)。冻结分类策略为中性logit加−0.4后取最大值：ℓ′=ℓ+(0,−0.4,0)，ĉ=argmax ℓ′。最终强度按类别保持符号一致：负向min(r,−10⁻⁶)，中性0，正向max(r,10⁻⁶)。所有主表MAE均评价该最终强度，同时另存raw_intensity。分类和回归共享全部编码及交互参数，输出层有各任务对应的行，联合训练；主模型未启用有序分类头，但保留强度的有序辅助损失。')
para('对每条样本，以预测类别ĉ与最强竞争类别c₂的校准logit差M=ℓ′ĉ−ℓ′c₂为分类解释目标。将所有4维读出投影到这一类别差后，记单模态总项Aₘ、交互总项Iₘₙ。每个成对交互的一半分配给相关两端，得到模态净分数与位置净分数：')
eq('Φₘ = Aₘ + ½ Σₙ≠ₘ Iₘₙ； ψₘⱼ = ãₘⱼ + ½ Σₙ≠ₘ ĩₘₙⱼ； M = βM + ΣₘΦₘ')
eq('ρₘ = |Φₘ| / Σₙ|Φₙ|； principal = argmaxₘ |Φₘ|')
para('ã和ĩ表示已缩放、已投影到解释目标的读出。βM包括两个对应输出偏置之差，以及中性−0.4校准项对该类别差的影响。解释卡同时保留符号：正值支持预测类别相对于竞争类别，负值反对该判断。ρ采用模态净值的绝对值归一化，偏置βM单列；它不同于把所有局部绝对值先相加得到的活性比例。若分母近零则标为未定义，主要模态不能强制指定。回归解释以q为目标同样核算，不能将其比例称为最终非线性强度的线性分配。动态可靠性与BERT上下文均随输入变化，因此当前分数项删除不等于重新运行被修改输入后的输出变化。')
head('5 模型求解与训练方案')
subhead('5.1 多任务目标函数')
eq('L = LHuber(r,y) + 0.5 LCE(ℓ,c) + 0.1 Lord + 0.001 Lpair\n+ 0.1 Luni + 0.1 Lneutral + 0.1 Lpolarity + 0.1 LNLL')
para('Huber阈值δ=1。Lpair为每样本所有已缩放成对项回归分量的绝对值之和，再在批内平均。Luni在实际可观测模态上平均Huber单模态回归损失与0.5倍单模态交叉熵；它使用uₘ的回归、分类分量。Lneutral使用中性logit减去正负logsumexp作为二分类log-odds；Lpolarity仅在非中性样本上对正负logit差进行二元交叉熵。Lord用固定带宽0.3、温度0.2构造两个累计logit：((r+0.3)/0.2, (r−0.3)/0.2)，分别以1[c>0]和1[c>1]为目标计算二元交叉熵并平均；该项不是启用独立的可学习有序头。')
eq('LNLL = mean有效模态{ ½[exp(−vₘ)(3 tanh(uₘ,₀)−y)² + vₘ] }')
para('式（9）在批内所有实际可观测的“样本—模态”对上取平均，记这些索引对的集合为O、数量为N_O；abs表示绝对值。i为样本索引，uᵢₘ,₀为对应单模态读出的回归分量。模态丢弃不改变此项的实际观测监督范围。')
para('异方差回归的NLL建模[4]用于单模态残差方差，使可靠性模块获得可监督训练信号。该损失不是三模态贡献的监督真值，预测方差的绝对大小不能直接等同于人类对证据可靠性的判断。整个模型用AdamW更新，BERT编码层和其他层使用固定分组学习率，当前实验没有学习率衰减或warmup。')
subhead('5.2 求解流程与关键参数')
para('求解分为五步：①在附件二训练集估计有效观测的标准化统计量；②载入公开预训练BERT，初始化局部编码、成对读出与方差模块；③按批次执行模态丢弃、联合前向与目标函数求导，使用AdamW更新参数；④每轮在验证集比较分类策略，保存三个选优目标对应的检查点；⑤选择固定检查点与决策策略，在测试集计算评价指标，在附件四输出预测及解释。附件四不参与上述训练和验证策略搜索。')
table(['参数','冻结主模型设置'],[['随机种子 / 批大小','17 / 32'],['BERT / 其他层学习率','2×10⁻⁵ / 3×10⁻⁴，固定'],['AdamW权重衰减 / 梯度裁剪','0.01（二维以上参数）/ 1.0'],['窗口 / 行投影 / 局部维度 / 交互秩','1 / 64 / 64 / 8'],['局部dropout / 模态丢弃率','0.2 / 0.1'],['最大轮次 / 早停耐心值','20 / 连续4轮三个选优目标均无刷新'],['实际轮数 / 采用检查点','9 / 第5轮'],['可训练BERT层 / AMP','12层，embedding和pooler冻结 / 关闭'],['总参数 / 可训练参数','109,555,007 / 85,127,231'],['分类最终策略','中性logit偏置−0.4；最大logit类别']], '主模型训练与推理参数')
para('每轮仅在验证集搜索策略：441组回归中性阈值和31个中性logit偏置，共472个候选；分别按Macro-F1、Accuracy及最终强度MAE维护独立最佳检查点。任一目标排序刷新即重置早停计数，不是只监测训练损失。本文主表统一取验证Macro-F1策略，不能混合不同轮次、不同偏置的最好指标。第5轮之后训练损失继续降低但验证原始MAE未刷新，支持保留早停；增加训练上限不自动增加有效训练轮数。')
figure('02_training','图2 主模型训练曲线。虚线为第5轮，实际第9轮停止。右图为原始强度MAE，区别于主表最终强度MAE。')
head('6 模型检验与误差分析')
subhead('6.1 评价指标与对比设置')
para('附件二验证集的正向样本占46.43%，负向和中性分别占28.30%和25.27%，类别分布不均衡。因此同时采用准确率、Macro-F1和中性F1评价分类，采用平均绝对误差MAE与Pearson相关系数评价强度。所有回归指标均以类别一致性处理后的最终强度计算。')
eq('Accuracy = N⁻¹ Σᵢ 1[ĉᵢ=cᵢ]； Macro-F1 = (F1负 + F1中 + F1正)/3')
eq('F1ₖ = 2TPₖ/(2TPₖ+FPₖ+FNₖ)； MAE = N⁻¹ Σᵢ |ŷᵢ−yᵢ|')
para('Pearson为真实强度与预测强度的样本相关系数，反映线性变化的一致程度；其数值不能替代绝对误差。分类指标按负向、中性、正向三类计算；不得与剔除中性的二分类成绩直接比较。')
para('选择两个具有已训练检查点的开源适配基线。SELF-MM[5]采用BERT文本编码、音视LSTM、融合及单模态读出和中心驱动动态伪标签；本实现增加比赛三分类读出、有效观测打包与数值稳定处理。TETFN[6]直接使用MMSA工具库[7]固定版本的网络源码，恢复对齐音视位置，增加比赛分类头及有界回归输出，并使用本地联合训练目标。二者均为比赛数据适配实验，不是原论文数据规模、训练目标和配置下的严格复现。')
para('所有方法仅以3395条训练样本学习，在同一728条验证集上选轮次和策略；本表统一使用各自验证Macro-F1版本。主模型最终按分类logit决策，而两基线的验证最优策略为回归阈值决策，策略差异一并列出，避免把结果误认为统一最后一层阈值的比较。')
subhead('6.2 基础性能与可视化分析')
labels={'uncertainty':'不确定性融合（主）','selfmm':'SELF-MM适配','tetfn_source':'TETFN源码适配'}
for split,title in [('valid','验证集（728条）'),('test','附件二测试集（727条）')]:
    table(['模型','Acc(%)','Macro-F1','MAE','Pearson','中性F1'],[[labels[x['family']],f"{100*float(x[split+'_accuracy']):.2f}",*[f"{float(x[split+'_'+k]):.4f}" for k in ['macro_f1','mae','pearson','neutral_f1']]] for x in CMP], title+'模型性能比较')
policies={}
for x in CMP:
    policy=json.loads(x['policy'])
    policies[x['family']]=(f"中性logit偏置{policy['neutral_bias']:+.1f}，取最大类别" if policy['decision_mode']=='classification' else f"回归双阈值：{policy['tau_minus']:.2f} / {policy['tau_plus']:.2f}")
table(['模型','检查点轮次','验证选定的最终策略'],[[labels[x['family']],x['epoch'],policies[x['family']]] for x in CMP], '模型选优轮次与决策策略')
para('主模型验证Macro-F1并非三者最高：SELF-MM为0.6390，主模型为0.6323。相对表中按验证Macro-F1选择的SELF-MM参考配置，主模型在附件2测试中准确率高2.48个百分点、Macro-F1高约0.0104，而MAE稍高约0.0040；相比表中TETFN参考配置，测试准确率、Macro-F1和MAE均更优。这一比较限定于表中配置；第6.5节进一步给出两类基线的其他参数结果。选择主模型同时考虑其原生可核算解释，不能称其在所有指标上领先。当前为单随机种子的代表配置比较，未给出多种子方差或统计显著性。')
figure('03_confusion','图3 三分类混淆矩阵，行是真值、列是预测。中性与正向之间的混淆较突出。')
table(['验证类别','Precision','Recall','F1','支持数'],[[CN[int(k)],*[f'{v[a]:.4f}' for a in ['precision','recall','f1']],v['support']] for k,v in MET['valid']['final']['per_class'].items()], '验证集逐类别性能')
figure('04_regression','图4 验证集最终强度散点与残差分布。预测中性置零，形成ŷ=0的水平带。')
subhead('6.3 原生结构对照与性能取舍')
para('在开源基线之外，补充原生骨干上“有序分类头”和“不确定性融合”两个模块的四格对照。四种设置均采用种子17、最多20轮、早停耐心4轮，以及相同的BERT学习率、其他层学习率、模态丢弃率和联合监督基础权重；各自按验证Macro-F1选择轮次和决策策略。有序分类头指可学习的累计阈值输出，与四种设置均保留的强度有序辅助损失不同。')
table(['结构设置','有序头','不确定性','轮次','Acc(%)','Macro-F1','MAE'],[[r['label'],'开' if r['ordinal_head'] else '关','开' if r['uncertainty_fusion'] else '关',r['epoch'],f"{100*r['valid']['accuracy']:.2f}",f"{r['valid']['macro_f1']:.4f}",f"{r['valid']['mae']:.4f}"] for r in SENS['structures']], '原生骨干的四种结构设置在验证集上的对照')
para('仅加入有序分类头时，验证Macro-F1为0.6230、MAE为0.5706；同时加入有序头与不确定性时，相应为0.6165和0.5688，均弱于本文主模型的0.6323和0.5397。对应的既有冻结测试结果中，两种有序设置的准确率均为66.02%，Macro-F1分别为0.6342和0.6212，MAE分别为0.6210和0.6184，也未优于主模型。由此可见，更强的输出约束或更多融合模块并不自动带来更好的分类与回归表现。')
para('联合监督基础设置在验证集的Macro-F1为0.6412，高于主模型；主模型的MAE则降低约0.0181。因此，四格对照支持不同结构之间存在性能取舍，不能据其中两个较弱设置宣称不确定性模块对所有指标均有增益。各设置使用各自验证最优策略，比较包含训练与策略选择的共同影响；单种子结果也不构成稳定提升的证据。')
subhead('6.4 学习率成组对照与较弱配置分析')
para('进一步从既有不确定性模型搜索中，提取两组完整的0.5倍、1倍、2倍学习率试验。A组采用本文主模型的监督权重与模态丢弃率；B组采用较强分类监督，分类、中性、极性、不确定性损失权重依次为4、1、2、1，模态丢弃率为0.05。两组单模态损失权重均为0.1。在每组内部，仅同时缩放BERT与其他层学习率，其余模型与训练参数保持一致；种子均为17、上限20轮、早停耐心6轮，每行仍取该配置自己的验证Macro-F1最优记录。')
table(['组别','倍率','BERT LR','其他层LR','轮次','Acc(%)','Macro-F1','MAE'],[[r['group'],f"{r['multiplier']:g}",f"{r['config']['train']['encoder_learning_rate']:.1e}",f"{r['config']['train']['learning_rate']:.1e}",r['epoch'],f"{100*r['valid']['accuracy']:.2f}",f"{r['valid']['macro_f1']:.4f}",f"{r['valid']['mae']:.4f}"] for r in SENS['learning_rates']], '不确定性模型的成组学习率对照（验证集）')
para('A组中，学习率减半或加倍均未改善参考配置的验证Macro-F1与MAE。B组的两组学习率从4×10⁻⁵/3×10⁻⁴同时加倍后，验证Macro-F1由0.6364降至0.6042，准确率由64.56%降至60.99%，MAE由0.5599增至0.5912，构成较明确的退化案例。该结果对应完整训练过程中的验证最优记录，并非从训练轨迹中抽取较差轮次；但它只能说明这组联合学习率设置不合适，不能分别归因于BERT或其他层的学习率。')
para('B组参考配置的验证Macro-F1高于A组，但MAE较大，也说明分类与回归目标之间仍有取舍。图6同时展示全部22个固定种子候选，完整参数表随结果提供；其余宽搜索点同时改变了多个超参数，不用于推断某个单独参数的因果作用。本节不额外使用测试集或附件四挑选较弱参数，也不将缺少对应冻结测试记录的配置填入测试成绩。')
figure('07_parameter_comparison','全部22个不确定性候选的验证性能与两组学习率对照。每点均使用本配置的验证Macro-F1最优轮次和策略；两组曲线连接的是成组参数试验，不是训练轮次。')
subhead('6.5 开源适配基线的参数对照')
para('进一步考察开源方法在比赛适配后的参数敏感性。纳入同批实验中固定种子17的全部6组SELF-MM适配配置和2组源码TETFN适配配置，包含较弱设置及各自参考设置。它们均使用附件二对齐版、批量32、最多10轮、早停耐心4轮，各自按验证Macro-F1确定检查点和决策策略。本节重载已有权重，先核对728条验证输出，再固定策略计算727条测试结果，不重新训练或使用测试集调参。该批预算与前述20轮上限实验不同，不能把10轮配置描述为已完成20轮训练。')
baseline_parameters=[]
for r in BASE:
    m,t=r['config']['model'],r['config']['train']
    baseline_parameters.append([r['code'],'SELF-MM' if r['family']=='selfmm' else 'TETFN',f"{t['encoder_learning_rate']:.0e}",
        m.get('post_fusion_dropout',m.get('dropout')),m['post_fusion_dim'],m.get('classification_weight',t['classification_weight']),f"{r['epoch']}/{r['actual_epochs']}"])
table(['代号','模型','BERT LR','Dropout','融合维数','分类权重','选中/实训'],baseline_parameters,'开源适配基线的参数设置（训练上限均为10轮）')
para('S0为无额外分类头的回归控制，S1为辅助分类监督，S2为温和辅助监督，S3为较宽融合层，S4为较强丢弃设置，S5为分组优化参考；T0、T1为两组源码TETFN配置。SELF-MM的Dropout作用于融合分支，TETFN该参数同时映射到多个网络丢弃位置，含义不能直接等同。S0、S5的音频/视觉/其他非BERT学习率分别为0.005、0.0001、0.001，BERT权重衰减为0.001、音视为0；S1、S3、S4非BERT学习率均为0.0003，S2为0.0002，其配置权重衰减为0.01。T0、T1非BERT学习率均为0.0003。完整配置与冻结策略见随附结果。')
table(['代号','验证Acc(%)','验证F1','验证MAE','测试Acc(%)','测试F1','测试MAE'],[[r['code'],*[f"{100*r[s]['accuracy']:.2f}" if k=='accuracy' else f"{r[s][k]:.4f}" for s in ['valid','test'] for k in ['accuracy','macro_f1','mae']]] for r in BASE], '各参数配置的验证与测试结果（F1均为Macro-F1）')
para('SELF-MM较弱参数的影响较明确：S3的验证Macro-F1为0.6227、测试Macro-F1为0.6187，测试MAE为0.6252；S5参考配置对应为0.6390、0.6389和0.5966。S0、S1、S4的测试Macro-F1分别为0.6300、0.6331、0.6298，亦低于S5。S4与S5的网络和BERT学习率相同，但非BERT分组学习率及权重衰减不同，因而这一比较体现的是优化配置的共同影响；S3还同时改变融合维数和BERT学习率，不能将其退化全部归因于“网络更宽”。')
para('完整对照也显示验证排序未必在测试中保持。S2的验证Macro-F1为0.6354，低于S5，但测试准确率69.60%、Macro-F1为0.6515，分别高于主模型的68.64%和0.6493；其测试MAE为0.6016，与主模型0.6006接近。T0的验证Macro-F1为0.6262，低于T1的0.6321，但测试Macro-F1反而为0.6513，高于T1的0.6285，测试MAE也更低。因此，T0只能称为验证指标较弱的设置，不能据此认定其测试性能较弱。')
para('S5与T1的验证、测试指标与第6.2节所列20轮上限参考实验一致，表明这两个参考设置在已有延长预算实验中未改变所选结果；其余设置仍受原10轮上限约束。本节结果支持参数选择和分类—回归之间存在取舍，也限制了“主模型全面优于开源基线”的说法。主模型的选用依据仍包括原生证据分解和题目解释要求；补充对照未据测试结果重新选择权重或修改模型。各行均为本比赛适配实验，不代表上游论文原始成绩。')
subhead('6.6 验证集错误归因')
para('验证集728条中分类错误254条。真实中性的184条中，89条被判为非中性（23条负向、66条正向），中性召回率51.63%；真实正向338条中67条被判中性，真实负向206条中48条被判中性。中性与正向混淆构成主要误差来源。中性边界校准改善取舍，仍不能解决说明性文本、弱情感与标注差异；继续扩大偏置会同时改变正负样本被归零的比例。')
vc={x['id']:x for x in rows('results/contributions/all_samples_contributions.csv') if x['split']=='valid'}
groups=[]
for name,pred in [('真实负向',lambda x:int(x['true_class'])==0),('真实中性',lambda x:int(x['true_class'])==1),('真实正向',lambda x:int(x['true_class'])==2),('视觉无有效观测',lambda x:vc[x['id']]['available_V']=='False'),('视觉存在有效观测',lambda x:vc[x['id']]['available_V']=='True')]:
    sub=[x for x in valid if pred(x)];error=sum(x['polarity']!=x['true_class'] for x in sub);mae=np.mean([abs(float(x['intensity'])-float(x['true_intensity'])) for x in sub]);groups.append([name,len(sub),error,f'{100*error/len(sub):.2f}%',f'{mae:.4f}'])
table(['验证分组','样本数','分类错误数','错误率','最终MAE'],groups, '验证集标签与观测分组的误差')
para('分组差异同时受标签分布、语句内容和观测质量影响，只能作为错误诊断线索，不构成模态缺失的因果效应。分类贡献在文本上的集中说明模型更依赖语言判别；语音与视觉的较低分类净贡献也可能来自内部正负项抵消，不能视为它们没有可用信息。强度预测的收缩与中性归零会对强情感样本产生较大残差，图5可用于检查这种系统偏差。')
subhead('6.7 解释完备性检验')
table(['解释检查','验证','测试','附件4'],[['分类间隔最大核算误差',*[f"{CON[s]['classification']['max_accounting_error']:.2e}" for s in ['valid','test','special']]],['回归q最大核算误差',*[f"{CON[s]['regression']['max_accounting_error']:.2e}" for s in ['valid','test','special']]],['逐样本核查数量',728,727,20]], '分类间隔与回归预激活的分解误差')
para('将三模态净分数与偏置相加，并与同次前向的目标分数逐条比较。1475条样本的分解误差均处于浮点计算误差范围。该检查说明解释忠实于当前读出的数值，不证明某个证据词是人类情感成因。本主模型没有单独完成随机扰动对照、多种子解释稳定性或人类证据标注评价，局部读出核算与输入扰动忠实性仍需分别检验。')
head('7 三模态作用差异与局部重要性')
subhead('7.1 分类、回归与可靠性的区别')
table(['集合/解释目标','文本(%)','语音(%)','视觉(%)'],[[{'valid':'验证集','test':'附件二测试集','special':'附件四'}[split]+'/'+target,*[f"{100*CON[split][key]['mean_shares'][m]:.2f}" for m in 'TAV']] for split in ['valid','test','special'] for key,target in [('classification','分类间隔'),('regression','回归q')]], '三模态净贡献占比的样本均值')
para('上述比例先在每条样本内归一化，再对样本等权平均。验证与测试的分类文本占比均约85%，而回归文本占比约68%—71%，语音和视觉对强度的作用相对更高。附件4分类主要参考模态为文本20/20，回归主要模态为文本18/20、视觉2/20（02、14）；02与14的视觉回归净贡献占比分别为48.12%与41.09%。因此“分类主要看文本”不能推导出“强度估计中的视觉无作用”。')
figure('05_contributions','图5 分类贡献、回归贡献与可靠性权重的对照。三者解释不同对象，不能互换。')
para('附件4平均可靠性权重T/A/V为45.07%/28.80%/26.12%，分类净贡献却为89.78%/4.71%/5.51%。这来自不同模态证据大小、方向和抵消程度的共同影响，表明直接把可靠性权重写成贡献会失真。附件4的13号视觉数组全部为零，视觉单项和交互关闭，其视觉贡献应为0；验证和测试分别有15、28条视觉无有效数值观测。')
subhead('7.2 主要参考模态内的局部证据分布')
para('局部重要性按分类间隔的有符号读出绘制。为避免只呈现支持结论的片段，图中保留每个有效位置及其正负方向：蓝色为支持预测类，红色为反对预测类。横轴采用原始分数量纲，纵轴保留原BERT位置与原文词元；这使典型解释可以回查到输入，而不只给出无位置的模态占比。')
figure('06_local_16','图6 附件4样本16的主要参考模态内局部重要性。每行给出原BERT位置和对应原文，负值表示反对当前类别间隔；词的上下文由BERT编码。')
figure('06_local_02','图7 附件4样本02的文本局部重要性。保留正负方向，绝对值大不等于支持当前预测。')
def evidence(sid,mod,limit=3):
    e=EXP[sid]; lookup={j:x for x in MEDIA[sid]['entries'] for j in x['source_indices']};lines=[]
    if mod=='V' and SHARE[sid]['available_V']=='False':return '无有效视觉观测；不列视觉证据。'
    for item in e['top_evidence'][mod][:limit]:
        if abs(item['signed_score'])<1e-10:continue
        entries=[lookup[j] for j in item['source_indices'] if j in lookup]
        word=' / '.join(dict.fromkeys(z['text'] for z in entries))
        spans=[span for z in entries for span in z.get('char_spans',[])]
        starts=[z.get('start_seconds') for z in entries if z.get('start_seconds') is not None];ends=[z.get('end_seconds') for z in entries if z.get('end_seconds') is not None]
        timing=f'；估计{min(starts):.2f}—{max(ends):.2f}s' if starts and ends else '；时间未定位'
        lines.append(f"位置{item['source_indices']}，{word}，字符{spans}，分数{item['signed_score']:+.4f}"+timing)
    return '；\n'.join(lines) or '无非零局部项。'

head('8 附件四全量预测与典型解释')
subhead('8.1 全量预测与三模态作用程度')
para('附件四按题目中的无标签专项集处理。表中给出全部20条样本的预测极性、最终强度、分类间隔的三模态净贡献占比及主要参考模态。比例均由同次前向的有符号净分数取绝对值后归一化；少量行因四舍五入可能不恰好合计100%。')
table(['编号','预测极性','最终强度','文本(%)','语音(%)','视觉(%)','主模态'],[[sid,CN[int(PRED[sid]['polarity'])],strength(PRED[sid]['intensity']),*[f"{100*float(SHARE[sid]['classification_modality_shares_'+m]):.2f}" for m in 'TAV'],SHARE[sid]['classification_principal_modality']] for sid in sorted(EXP)], '附件四20条样本的预测与分类贡献汇总')
counts=[sum(int(r['polarity'])==k for r in PRED.values()) for k in range(3)]
para(f'模型预测的类别分布为负向{counts[0]}条、中性{counts[1]}条、正向{counts[2]}条。该分布是预测输出的统计。样本02的最终强度为+1.0×10⁻⁶，来自类别一致性处理，故以科学计数法保留其正号，避免将其显示为中性0。')
subhead('8.2 关键证据定位与结果文件')
para('对每条样本，按局部贡献绝对值选取关键项，同时保留其支持或反对的方向。表中列主要参考模态内最显著的局部词元及原序列索引；完整结果文件同时给出三模态关键证据、原文本字符区间及估计时间。表内的语音时间来自MMS强制对齐[8]，仅用于媒体回查；同位置视觉时间为粗略检索区间，不视为原视觉特征的精确帧时间。')
evidence_rows=[]
for sid in sorted(EXP):
    top=EXP[sid]['top_evidence'][EXP[sid]['principal_modality']][0]
    lookup={j:z for z in MEDIA[sid]['entries'] for j in z['source_indices']}
    entries=[lookup[j] for j in top['source_indices'] if j in lookup]
    words=' / '.join(dict.fromkeys(z['text'] for z in entries))
    starts=[z['start_seconds'] for z in entries if z.get('start_seconds') is not None]
    ends=[z['end_seconds'] for z in entries if z.get('end_seconds') is not None]
    timing=f'{min(starts):.2f}—{max(ends):.2f}' if starts and ends else '未定位'
    evidence_rows.append([sid,f"{top['source_indices']} / {words}",f"{top['signed_score']:+.4f}",timing])
table(['编号','主模态关键位置 / 原词元','有符号贡献','估计时段(s)'], evidence_rows, '附件四20条样本的主要证据定位索引')
para('完整预测与解释以CSV汇总：一行对应一个样本，含类别、最终及原始强度、分类与回归净分数及比例、主要模态、三模态关键证据和可用状态。全部20条详细解释卡与该表使用同一批固定预测记录生成。')
subhead('8.3 典型样本解释卡')
para('选取02、13、14、16号样本，分别展示分类与回归分歧、视觉缺失、中性决策以及文本高度主导的读出。每张卡同时列预测、三模态作用程度、主要模态、证据位置及数值方向；其解读仅说明模型如何形成当前判断。字符区间采用原文本的半开区间，序列位置为零基BERT索引。')
notes={
'02':'分类主要参考文本，回归q主要参考视觉，两项任务的主要模态不同。原始强度为负而分类判为正，最终强度经一致性处理成为接近零的正值；这是当前双任务输出间的分歧，不应把最终微小正值写成强正向判断。文本中支持项与反对项同时存在，词面含义不能替代上下文化读出。',
'13':'原视觉数组全零，视觉单项及关联交互关闭，视觉贡献为0。当前判断来自可观测的文本与语音，分类输出为中性。缺失是输入的观测事实，不能由这一案例推断若补充视觉就会改变类别。',
'14':'分类判为中性，最终强度置零；回归q中视觉净作用占41.09%，高于其他单个模态。中性类别并不意味着各模态没有作用，局部支持、反对以及输出偏置共同决定最终分类间隔。',
'16':'预测为负向，最终强度为−2.6363。文本净贡献占99.04%，原句中两处terrible可在局部图中回查；这些读出已包含全句上下文，不能解释为词自身独立产生的因果效应。'}
card_lines=['# 附件四全20条解释卡','本文件按无标签专项集展示预测与解释。秒数为机器对齐估计，视觉检索时段并非原特征精确帧时间。T/A/V分别表示文本/语音/视觉；有符号分数针对预测类与竞争类的间隔，正值支持、负值反对。']
formal_rows=[]
for sid in sorted(EXP):
    s=SHARE[sid];r=PRED[sid];e=EXP[sid]
    competitor=int(e['target'].rsplit('_',1)[-1])
    data=[['预测极性 / 最终强度',f"{CN[int(r['polarity'])]} / {strength(r['intensity'])}"],
          ['原始强度 / 竞争类别',f"{strength(s['raw_intensity'])} / {CN[competitor]}"],
          ['分类贡献 T / A / V',' / '.join(f"{100*float(s['classification_modality_shares_'+m]):.2f}%" for m in 'TAV')],
          ['分类净分数 T / A / V',' / '.join(f"{float(s['classification_modal_scores_'+m]):+.4f}" for m in 'TAV')],
          ['回归q贡献 T / A / V',' / '.join(f"{100*float(s['regression_modality_shares_'+m]):.2f}%" for m in 'TAV')],
          ['分类 / 回归主要模态',s['classification_principal_modality']+' / '+s['regression_principal_modality']],
          ['分类间隔 / 偏置',f"{e['score']:.4f} / {e['bias']:+.4f}"],['原文本',MEDIA[sid]['raw_text']]]
    card_lines+=['',f'## 样本{sid}']+[f'**{a}：** {b}' for a,b in data]
    for m in 'TAV': card_lines+=[f'**{m}关键证据：** '+evidence(sid,m)]
    if sid in notes:
        card_lines+=['**解读：** '+notes[sid]]
        add('card_start','样本'+sid)
        table(['项目','内容'],data, '样本'+sid+'解释卡')
        for m in 'TAV': para(m+'关键证据：'+evidence(sid,m,limit=3 if m=='T' else 1))
        para('解读：'+notes[sid])
    out={'id':sid,'predicted_polarity':int(r['polarity']),'predicted_label':CN[int(r['polarity'])],
         'predicted_intensity':r['intensity'],'raw_intensity':s['raw_intensity'],
         'classification_principal_modality':s['classification_principal_modality'],
         'regression_principal_modality':s['regression_principal_modality'],
         'classification_margin':e['score'],'classification_bias':e['bias'],
         'classification_accounting_error':e['accounting_error'],'raw_text':MEDIA[sid]['raw_text']}
    for m in 'TAV':
        for key in ('classification_modal_scores','classification_modality_shares','regression_modal_scores','regression_modality_shares','precision_weight','available'):
            out[key+'_'+m]=s[key+'_'+m]
        out['key_evidence_'+m]=evidence(sid,m)
    out['time_note']='机器对齐估计；视觉时段仅用于回查，并非原特征精确帧时间'
    formal_rows.append(out)
(ROOT/'results/附件4_全20条解释卡.md').write_text('\n\n'.join(card_lines)+'\n')
with (ROOT/'results/附件4_预测与解释汇总.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(formal_rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(formal_rows)
assert len(formal_rows)==20 and {r['id'] for r in formal_rows}=={f'{i:02d}' for i in range(1,21)}
assert all(not re.search(r'true|ground|sentiment|accuracy|mae|pearson|f1',k,re.I) for k in formal_rows[0])
head('9 模型评价与改进方向')
subhead('9.1 模型特点与适用性')
para('本模型将预测与解释建立在同一组实际读出上。局部项保留原输入索引，单项和成对项能够逐层汇总到模态贡献，便于满足题目要求的“预测—模态—片段”解释链条。可靠性模块仅增加27个参数，不需要为每条解释重复训练模型；主要计算开销仍来自BERT。分类间隔与回归预激活分别分析，也避免了用单一贡献比例概括两项不同任务。')
subhead('9.2 模型局限')
para('首先，有限训练数据与多次验证选择带来过拟合和选择偏差，当前比较只有一个随机种子，未给出多次重复的方差。其次，分类文本贡献较高，中性与弱极性边界仍有明显混淆；较低音视净贡献可能由证据弱或内部抵消造成，不能仅凭比例判断其信息无用。再次，可加读出保证的是当前前向的数值完备性，不是独立词效应或输入删除的因果忠实性；方差针对单模态回归残差，未验证为融合后的概率区间。最后，音视精确时间戳缺失限制了证据定位精度，三方高阶交互也未显式建模。')
subhead('9.3 后续改进方向')
para('后续优先扩展本次单种子结构与学习率对照，进行多随机种子重复和独立确认实验，并进一步分离成对项、辅助监督等模块的作用；围绕中性边界采用嵌套验证控制策略选择偏差，分别考察分类、回归与校准质量。在解释层面，可增加保持训练数据分布的局部扰动实验、跨种子稳定性评价和人工证据核验，区分数值完备、预测忠实与人类可理解三个维度。上述扩展内容为待验证方向，不计入本文已完成结果。')
head('参考文献')
for reference in [
'[1] Devlin J, Chang M W, Lee K, Toutanova K. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL-HLT, 2019: 4171–4186. https://aclanthology.org/N19-1423/',
'[2] Agarwal R, Melnick L, Frosst N, et al. Neural Additive Models: Interpretable Machine Learning with Neural Nets. NeurIPS, 2021, 34: 4699–4711. https://arxiv.org/abs/2004.13912',
'[3] Lou Y, Caruana R, Gehrke J, Hooker G. Accurate Intelligible Models with Pairwise Interactions. KDD, 2013: 623–631. https://doi.org/10.1145/2487575.2487579',
'[4] Kendall A, Gal Y. What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? NeurIPS, 2017, 30. https://arxiv.org/abs/1703.04977',
'[5] Yu W, Xu H, Yuan Z, Wu J. Learning Modality-Specific Representations with Self-Supervised Multi-Task Learning for Multimodal Sentiment Analysis. AAAI, 2021, 35(12): 10790–10797. https://arxiv.org/abs/2102.04830',
'[6] Wang D, Guo X, Tian Y, Liu J, He L, Luo X. TETFN: A Text Enhanced Transformer Fusion Network for Multimodal Sentiment Analysis. Pattern Recognition, 2023, 136: 109259. https://doi.org/10.1016/j.patcog.2022.109259',
'[7] thuiar. MMSA: Multimodal Sentiment Analysis Toolkit[EB/OL]. https://github.com/thuiar/MMSA, 访问日期：2026-09-26. 本文采用固定源码版本a94e65d的比赛适配实现。',
'[8] Pratap V, Tjandra A, Shi B, et al. Scaling Speech Technology to 1,000+ Languages. Journal of Machine Learning Research, 2024, 25(97): 1–52. https://jmlr.org/papers/v25/23-1318.html.'
]: add('reference',reference)

# Render a single-column, anonymous mathematical modelling manuscript.
# Equations use editable Word math runs; captions/tables share the Markdown source.
def set_font(style,east='宋体',size=12):
    style.font.name='Times New Roman';style.font.size=Pt(size)
    style._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),east)

doc=Document();sec=doc.sections[0]
sec.page_height=Cm(29.7);sec.page_width=Cm(21)
sec.top_margin=Cm(2.2);sec.bottom_margin=Cm(2.0);sec.left_margin=Cm(2.3);sec.right_margin=Cm(2.3)
sec.footer_distance=Cm(1.0)
normal=doc.styles['Normal'];set_font(normal)
normal.paragraph_format.line_spacing=1.0;normal.paragraph_format.space_after=Pt(6)
normal.paragraph_format.first_line_indent=Pt(24)
normal.paragraph_format.widow_control=True
for name,size in [('Title',16),('Heading 1',14),('Heading 2',12)]:
    style=doc.styles[name];set_font(style,'黑体',size);style.font.color.rgb=RGBColor(0,0,0)
    style.font.bold=True;style.paragraph_format.first_line_indent=Pt(0)
    style.paragraph_format.space_before=Pt(12);style.paragraph_format.space_after=Pt(8)
    style.paragraph_format.keep_with_next=True
    style.paragraph_format.alignment=1 if name!='Heading 2' else 0
set_font(doc.styles['Caption'],size=10.5)
doc.styles['Caption'].font.italic=False
doc.styles['Caption'].font.color.rgb=RGBColor(0,0,0)
doc.styles['Caption'].paragraph_format.first_line_indent=Pt(0)
doc.styles['Caption'].paragraph_format.alignment=1
doc.core_properties.author='';doc.core_properties.last_modified_by=''
doc.core_properties.title='基于不确定性融合与局部可加证据的多模态情感识别'
doc.core_properties.subject='问题三：模型、检验、预测与解释'
doc.core_properties.comments=''
footer=sec.footer.paragraphs[0];footer.alignment=1;footer.paragraph_format.first_line_indent=Pt(0)
field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field)
# Explicitly start numbering at the abstract page, with no running header.
page_num=OxmlElement('w:pgNumType');page_num.set(qn('w:start'),'1');sec._sectPr.append(page_num)

def caption(text,keep=False):
    p=doc.add_paragraph(text,'Caption');p.paragraph_format.keep_with_next=keep
    return p

def no_indent(p):p.paragraph_format.first_line_indent=Pt(0)

# Real subscripts/superscripts avoid unsupported Unicode small-letter glyphs.
SUB=dict(zip('₀₁₂₃₄₅₆₇₈₉ₘₙⱼᵢₖ₍₎','0123456789mnjik()'))
SUP=dict(zip('⁰¹²³⁴⁵⁶⁷⁸⁹⁻ᵀ','0123456789-T'))
def add_text(p,value):
    for fragment in re.findall(r'[₀-₉ₘₙⱼᵢₖ₍₎]+|[⁰¹²³⁴⁵⁶⁷⁸⁹⁻ᵀ]+|[^₀-₉ₘₙⱼᵢₖ₍₎⁰¹²³⁴⁵⁶⁷⁸⁹⁻ᵀ]+',value):
        mapping=SUB if fragment[0] in SUB else SUP if fragment[0] in SUP else None
        run=p.add_run(''.join(mapping[c] for c in fragment) if mapping else fragment)
        if mapping is SUB:run.font.subscript=True
        elif mapping is SUP:run.font.superscript=True

def math_nodes(node):
    """Map the small MathML vocabulary used here to native Word math objects.

    Literal delimiters and script sums keep the equations readable in both
    Word and older LibreOffice versions, without empty operator placeholders.
    """
    tag=etree.QName(node).localname
    def run(text,italic=False):
        r=OxmlElement('m:r');pr=OxmlElement('m:rPr');style=OxmlElement('m:sty')
        style.set(qn('m:val'),'i' if italic else 'p');pr.append(style);r.append(pr)
        t=OxmlElement('m:t');t.text=text;r.append(t);return r
    if tag in ('math','mrow','mstyle'):
        return [part for child in node for part in math_nodes(child)]
    if tag in ('mi','mn','mo','mtext'):
        value=node.text or ''
        return [run(value,tag=='mi' and len(value)==1 and node.get('mathvariant') not in ('normal','sans-serif'))]
    if tag=='mspace':return [run(' ')]
    shapes={'msub':('sSub',['e','sub']),'msup':('sSup',['e','sup']),
            'msubsup':('sSubSup',['e','sub','sup']),'mfrac':('f',['num','den']),
            'munder':('sSub',['e','sub']),'munderover':('sSubSup',['e','sub','sup'])}
    if tag in shapes:
        shape,slots=shapes[tag];out=OxmlElement('m:'+shape)
        for child,slot in zip(node,slots):
            holder=OxmlElement('m:'+slot)
            for part in math_nodes(child):holder.append(part)
            out.append(holder)
        return [out]
    if tag=='msqrt':
        out=OxmlElement('m:rad');pr=OxmlElement('m:radPr');hide=OxmlElement('m:degHide');hide.set(qn('m:val'),'1');pr.append(hide);out.append(pr)
        out.append(OxmlElement('m:deg'));holder=OxmlElement('m:e')
        for child in node:
            for part in math_nodes(child):holder.append(part)
        out.append(holder);return [out]
    if tag=='mover':
        out=OxmlElement('m:acc');pr=OxmlElement('m:accPr');char=OxmlElement('m:chr')
        accent=''.join(node[1].itertext());char.set(qn('m:val'),{'~':'̃','^':'̂'}.get(accent,accent));pr.append(char);out.append(pr)
        holder=OxmlElement('m:e')
        for part in math_nodes(node[0]):holder.append(part)
        out.append(holder);return [out]
    raise ValueError(f'Unsupported equation element: {tag}')

LATEX=[
 [r'd_{mj}=E_m(x_{mj},o_{mj})-E_m(0,o_{mj})'],
 [r'a_{mj}=\omega_{mj}W_md_{mj},\quad i_{mnj}=\omega_{mnj}V_{mn}[(P_{mn}d_{mj})\odot(Q_{mn}d_{nj})]'],
 [r'v_m=-1+4\tanh(w_m^{\mathsf T}[u_m;\sum_j\mathrm{abs}(a_{mj})]+b_m),\quad \sigma_m^2=\exp(v_m)'],
 [r'p_m=\frac{\mathbf1_m\exp(-v_m)}{\sum_n\mathbf1_n\exp(-v_n)},\quad \kappa_m=Kp_m,\quad K=\sum_m\mathbf1_m'],
 [r's=\beta+\sum_{m,j}\kappa_m a_{mj}+\sum_{(m,n)\in\mathcal P,j}\sqrt{\kappa_m\kappa_n}\,i_{mnj}',r'q=s_0,\quad \ell=(s_1,s_2,s_3)'],
 [r'\Phi_m=A_m+\frac12\sum_{n\ne m}I_{mn},\quad\psi_{mj}=\tilde{a}_{mj}+\frac12\sum_{n\ne m}\tilde{i}_{mnj}',r'M=\beta_M+\sum_m\Phi_m'],
 [r'\rho_m=\frac{\mathrm{abs}(\Phi_m)}{\sum_n\mathrm{abs}(\Phi_n)},\quad m_{\mathrm{ref}}=\mathrm{argmax}_m\,\mathrm{abs}(\Phi_m)'],
 [r'L=L_{\mathrm{Huber}}(r,y)+0.5L_{\mathrm{CE}}(\ell,c)+0.1L_{\mathrm{ord}}+0.001L_{\mathrm{pair}}',r'\quad+0.1L_{\mathrm{uni}}+0.1L_{\mathrm{neutral}}+0.1L_{\mathrm{polarity}}+0.1L_{\mathrm{NLL}}'],
 [r'e_{im}=3\tanh(u_{im0})-y_i',r'L_{\mathrm{NLL}}=\frac{1}{2N_{\mathcal O}}\sum_{(i,m)\in\mathcal O}(\exp(-v_{im})e_{im}^2+v_{im})'],
 [r'\mathrm{Accuracy}=\frac1N\sum_i\mathbf1[\hat c_i=c_i],\quad \mathrm{MacroF1}=\frac13\sum_{k=0}^2\mathrm{F1}_k'],
 [r'\mathrm{F1}_k=\frac{2\mathrm{TP}_k}{2\mathrm{TP}_k+\mathrm{FP}_k+\mathrm{FN}_k},\quad \mathrm{MAE}=\frac1N\sum_i\mathrm{abs}(\hat y_i-y_i)']
]

md=[];table_number=figure_number=eq_number=card_number=0
for kind,value in blocks:
    if kind=='title':doc.add_heading(value,0);md+=['# '+value,'']
    elif kind=='subtitle':
        p=doc.add_paragraph(value);p.alignment=1;no_indent(p);md+=[value,'']
    elif kind=='abstract_title':
        p=doc.add_paragraph();p.alignment=1;no_indent(p);p.add_run(value).bold=True;md+=['## 摘要','']
    elif kind=='keywords':
        p=doc.add_paragraph(value);no_indent(p);p.runs[0].bold=True;md+=[value,'']
    elif kind=='pagebreak':doc.add_page_break()
    elif kind in ['h','h2']:
        p=doc.add_heading(value,1 if kind=='h' else 2)
        if kind=='h' and value.startswith(('8 ','9 ')):p.paragraph_format.page_break_before=True
        md+=[('## ' if kind=='h' else '### ')+value,'']
    elif kind=='card_start':
        p=doc.add_heading(value,2);p.paragraph_format.page_break_before=card_number>0;card_number+=1;md+=['### '+value,'']
    elif kind=='p':add_text(doc.add_paragraph(),value);md+=[value,'']
    elif kind=='reference':
        p=doc.add_paragraph(value);no_indent(p);p.paragraph_format.line_spacing=1.0
        for run in p.runs:run.font.size=Pt(10.5)
        md+=[value,'']
    elif kind=='eq':
        eq_number+=1
        for line_no,line in enumerate(LATEX[eq_number-1]):
            p=doc.add_paragraph();no_indent(p);p.alignment=1
            p.paragraph_format.keep_with_next=line_no<len(LATEX[eq_number-1])-1
            p.paragraph_format.space_after=Pt(4)
            formula=OxmlElement('m:oMath')
            for part in math_nodes(etree.fromstring(latex_to_mathml(line).encode())):formula.append(part)
            p._p.append(formula)
            if line_no==len(LATEX[eq_number-1])-1:p.add_run(f'　（{eq_number}）')
        md+=[value.replace('\n','<br>')+f'　（{eq_number}）','']
    elif kind=='figure':
        figure_number+=1;name,description=value
        p=doc.add_paragraph();no_indent(p);p.alignment=1
        p.paragraph_format.keep_with_next=True
        p.add_run().add_picture(str(FIG/(name+'.png')),width=Cm(16.1))
        title=f'图{figure_number}　{description}';caption(title)
        md += [f'![{title}](figures/{name}.png)','']
    elif kind=='table':
        table_number+=1;headers,data,description=value
        title=f'表{table_number}　{description}';caption(title,keep=True)
        t=doc.add_table(rows=1,cols=len(headers));t.autofit=False
        t.alignment=1
        if len(headers)==2:
            widths=[4.1,12.3]
        elif len(headers)==4 and headers[0]=='编号':widths=[1.4,7.4,3.3,4.3]
        elif len(headers)==6 and headers[0]=='集合':widths=[3.3,1.5,1.4,1.4,1.4,7.4]
        elif len(headers)==6 and headers[0]=='模型':widths=[4.4,2.2,2.8,2.2,2.6,2.6]
        elif len(headers)==4:widths=[6.2,3.4,3.4,3.4]
        elif len(headers)==3:widths=[4.5,3.0,8.9]
        elif len(headers)==7 and headers[0]=='结构设置':widths=[4.1,1.6,1.9,1.4,2.2,2.8,2.2]
        elif len(headers)==7 and headers[:2]==['代号','模型']:widths=[1.1,2.5,2.5,2.1,2.3,2.4,2.5]
        elif len(headers)==7 and headers[0]=='代号':widths=[1.5,2.65,2.4,2.3,2.65,2.4,2.3]
        elif len(headers)==8:widths=[1.2,1.2,2.5,2.5,1.1,2.2,2.8,2.3]
        elif len(headers)==7:widths=[1.2,2.1,3.0,2.45,2.45,2.45,2.75]
        else:widths=[16.4/len(headers)]*len(headers)
        widths=[w*16.4/sum(widths) for w in widths]
        for col,w in zip(t.columns,widths):col.width=Cm(w)
        for cell,text in zip(t.rows[0].cells,headers):add_text(cell.paragraphs[0],str(text))
        repeat=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(repeat)
        for row in data:
            for cell,text in zip(t.add_row().cells,row):add_text(cell.paragraphs[0],str(text))
        # Three-line table: outer top/bottom and one line below the header.
        props=t._tbl.tblPr;borders=OxmlElement('w:tblBorders')
        for edge in ['top','left','bottom','right','insideH','insideV']:
            border=OxmlElement('w:'+edge);border.set(qn('w:val'),'single' if edge in ['top','bottom'] else 'nil')
            border.set(qn('w:sz'),'10');border.set(qn('w:color'),'000000');borders.append(border)
        props.append(borders)
        for ri,row in enumerate(t.rows):
            row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
            for cell,w in zip(row.cells,widths):
                cell.width=Cm(w);cell.vertical_alignment=1
                if ri==0:
                    tc_b=OxmlElement('w:tcBorders');line=OxmlElement('w:bottom');line.set(qn('w:val'),'single');line.set(qn('w:sz'),'6');tc_b.append(line);cell._tc.get_or_add_tcPr().append(tc_b)
                for p in cell.paragraphs:
                    no_indent(p);p.paragraph_format.space_after=Pt(3);p.paragraph_format.space_before=Pt(3);p.paragraph_format.line_spacing=1.0
                    if len(data)<=12:p.paragraph_format.keep_with_next=ri<len(t.rows)-1
                    p.alignment=0 if len(headers)==2 else 1
                    for run in p.runs:run.font.size=Pt(10.5);run.bold=(ri==0)
        md+=[title,'','| '+' | '.join(map(str,headers))+' |','|'+'|'.join(['---']*len(headers))+'|']
        md+=['| '+' | '.join(str(x).replace('\n','<br>').replace('|','/') for x in row)+' |' for row in data]+['']
path=PAPER/'问题三_不确定性融合论文'
doc.save(path.with_suffix('.docx'));path.with_suffix('.md').write_text('\n'.join(md)+'\n')
print(json.dumps({'paper':str(path),'figures':figure_number,'tables':table_number,'equations':eq_number,'cards':len(EXP)},ensure_ascii=False))
