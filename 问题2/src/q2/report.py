"""Evidence-first report data and scientific figures; no training is performed."""
import csv
import json
from pathlib import Path
import numpy as np
from .data import CLASSES, digest, load_data, load_pickle, save_json
from .missing import MAIN, SUPPLEMENT


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def table(headers, rows):
    def cell(x):
        if x is None:
            return "未定义"
        if isinstance(x, float):
            return f"{x:.4f}"
        return str(x).replace("|", "/").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(cell, headers)) + " |",
                       "| " + " | ".join(["---"] * len(headers)) + " |"] +
                      ["| " + " | ".join(map(cell, row)) + " |" for row in rows])


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(root):
    root = Path(root)
    out = root / "reports"
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    frozen = read_json(root / "study/frozen.json")
    selected = frozen["selected_run"]
    search, stability = read_json(root / "study/search.json"), read_json(root / "study/stability.json")
    audit, environment = read_json(root / "audit/data_audit.json"), read_json(root / "audit/environment.json")
    cfg = read_json(root / "runs" / selected / "config.json")
    run_names = ["mvp_clean_s42", "mvp_span_s42"] + [x["run"] for x in search["candidates"] if x["status"] == "completed"]
    summaries, metric_rows = [], []
    for run in run_names:
        result = read_json(root / "runs" / run / "metrics_valid.json")
        metadata = read_json(root / "runs" / run / "metadata.json")
        summaries.append({"run": run, "clean_accuracy": result["clean"]["accuracy"],
                          "clean_f1": result["clean"]["macro_f1"], "clean_mae": result["clean"]["mae"],
                          **result["selection"], "best_epoch": metadata["best_epoch"],
                          "training_seconds": metadata["elapsed_seconds"]})
        for scenario in ["clean"] + MAIN:
            v = result[scenario]
            metric_rows.append(dict(run=run, split="valid", scenario=scenario, accuracy=v["accuracy"],
                                    macro_f1=v["macro_f1"], mae=v["mae"], pearson=v["pearson"],
                                    negative_f1=v["class_f1"][0], neutral_f1=v["class_f1"][1], positive_f1=v["class_f1"][2]))
    write_csv(out / "experiment_summary.csv", summaries)
    extended = {}
    for split in ["valid", "test"]:
        extended[split] = read_json(root / "runs" / selected / f"metrics_{split}_extended.json")
        for scenario, v in extended[split].items():
            if scenario == "selection":
                continue
            metric_rows.append(dict(run=selected, split=split, scenario=scenario, accuracy=v["accuracy"],
                                    macro_f1=v["macro_f1"], mae=v["mae"], pearson=v["pearson"],
                                    negative_f1=v["class_f1"][0], neutral_f1=v["class_f1"][1], positive_f1=v["class_f1"][2]))
    # Deduplicate the selected run's main validation rows from the extended export.
    metric_rows = list({(r["run"], r["split"], r["scenario"]): r for r in metric_rows}.values())
    write_csv(out / "all_scenario_metrics.csv", metric_rows)
    stability_rows = []
    for rec in stability["records"]:
        for scenario in ["clean"] + MAIN:
            v = rec["metrics"][scenario]
            stability_rows.append(dict(role=rec["role"], seed=rec["seed"], run=rec["run"], scenario=scenario,
                                       macro_f1=v["macro_f1"], mae=v["mae"], accuracy=v["accuracy"], pearson=v["pearson"]))
    write_csv(out / "stability_scenarios.csv", stability_rows)
    with (root / "runs" / selected / "predictions/valid/clean.csv").open(encoding="utf-8-sig") as f:
        predictions = list(csv.DictReader(f))
    errors = sorted(predictions, key=lambda r: abs(float(r["intensity"]) - float(r["true_intensity"])), reverse=True)[:10]
    raw = load_pickle(root / "data/raw/aligned_50.pkl")
    texts = {str(i): str(t) for i, t in zip(raw["valid"]["id"], raw["valid"]["raw_text"])}
    for row in errors:
        row["absolute_error"] = abs(float(row["intensity"]) - float(row["true_intensity"]))
        row["raw_text"] = texts[row["sample_id"]]
    write_csv(out / "largest_valid_regression_errors.csv", errors)
    groups = {s: set(str(x).split("$_$")[0] for x in raw[s]["id"]) for s in ["train", "valid", "test"]}
    overlaps = {f"{a}_{b}": len(groups[a] & groups[b]) for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]}
    disagreement = sum(int(r["class_id"]) != (0 if float(r["intensity"]) < 0 else 2 if float(r["intensity"]) > 0 else 1) for r in predictions)
    save_json(out / "error_analysis.json", {"classification_regression_sign_disagreements": disagreement,
              "validation_count": len(predictions), "video_group_overlap": overlaps,
              "note": "Two independent heads are not forced to agree. Regression near zero is not a calibrated neutral-class threshold. Official split retained; video overlap is a limitation, not an authorization to repartition."})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), layout="constrained")
    for modality, color in zip(["text", "audio", "vision"], ["#245A81", "#CE7735", "#447754"]):
        x = [0, 10, 30, 50]
        vals = [extended["valid"]["clean"]] + [extended["valid"][f"{modality}_{r}_random"] for r in x[1:]]
        axes[0].plot(x, [v["macro_f1"] for v in vals], marker="o", color=color, label=modality)
        axes[1].plot(x, [v["mae"] for v in vals], marker="o", color=color, label=modality)
    for ax, ylabel in zip(axes, ["Macro-F1", "MAE"]):
        ax.set(xlabel="Artificially removed valid positions (%)", ylabel=ylabel)
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
    fig.savefig(figures / "missing_impact.png", dpi=180)
    plt.close(fig)
    cm = np.array(extended["valid"]["clean"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.5, 3.5), layout="constrained")
    ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i,j] > cm.max()*.6 else "black")
    ax.set(xticks=range(3), yticks=range(3), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Predicted class", ylabel="True class")
    fig.savefig(figures / "confusion_valid.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.5, 3.3), layout="constrained")
    for run, color in [("mvp_span_s42", "#888888"), (selected, "#245A81")]:
        with (root / "runs" / run / "epochs.csv").open() as f:
            rows = list(csv.DictReader(f))
        ax.plot([int(r["epoch"]) for r in rows], [float(r["valid_score"]) for r in rows], label=run, color=color)
    ax.set(xlabel="Epoch", ylabel="Validation missing-scenario score")
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    fig.savefig(figures / "learning_curve.png", dpi=180)
    plt.close(fig)
    with (root / "final/attachment3_predictions.csv").open(encoding="utf-8-sig") as f:
        special = list(csv.DictReader(f))
    solution = build_markdown(root, selected, cfg, summaries, extended, stability, search, audit, environment, special, overlaps, disagreement)
    (out / "E题问题2解决方案与实验报告.md").write_text(solution, encoding="utf-8")
    save_json(out / "report_data.json", {"selected": selected, "config": cfg, "summaries": summaries,
              "metrics": extended, "stability": stability, "search": search, "special": special,
              "source_files": {"frozen": digest(root / "study/frozen.json"),
                               "metrics_valid": digest(root / "runs" / selected / "metrics_valid_extended.json"),
                               "metrics_test": digest(root / "runs" / selected / "metrics_test_extended.json")}})
    save_json(out / "figure_sources.json", {"missing_impact.png": "all_scenario_metrics.csv: selected model, valid, clean and nine random-span scenarios",
              "confusion_valid.png": f"runs/{selected}/metrics_valid_extended.json: clean.confusion_matrix",
              "learning_curve.png": f"runs/mvp_span_s42/epochs.csv and runs/{selected}/epochs.csv"})
    print(solution[:1500])


def build_markdown(root, selected, cfg, summaries, extended, stability, search, audit, environment, special, overlaps, disagreement):
    valid, test = extended["valid"], extended["test"]
    worst_f1 = min(MAIN, key=lambda s: valid[s]["macro_f1"])
    worst_mae = max(MAIN, key=lambda s: valid[s]["mae"])
    comparisons = []
    for run in dict.fromkeys(["mvp_clean_s42", "mvp_span_s42", selected]):
        t = read_json(root / "runs" / run / "metrics_test_extended.json")
        comparisons.append([run.replace('_s42',''),t['clean']['macro_f1'],t['clean']['mae'],t['selection']['mean_missing_macro_f1'],t['selection']['mean_missing_mae']])
    conclusion = "候选模型通过预先固定的三种子稳定性规则，因此作为最终模型。" if stability["stable_improvement"] else "候选模型未通过预先固定的三种子稳定性规则，最终保留连续缺失训练基线。"
    pieces = ["# E题问题2解决方案与实验报告", "## 1 结果与交付范围",
        f"本报告仅解决模态局部缺失条件下的情感极性与强度预测。最终模型为 `{selected}`。{conclusion}验证集完整输入 Macro-F1 为 {valid['clean']['macro_f1']:.4f}，MAE 为 {valid['clean']['mae']:.4f}；冻结后测试集对应结果为 {test['clean']['macro_f1']:.4f} 和 {test['clean']['mae']:.4f}。附件3共30条预测已生成。这里的完整输入指不额外施加人工缺失，原文件自身的零向量仍按缺失处理。",
        "本次完成可行版、有限候选比较、配对随机种子验证、冻结后测试和可复现交付。需要强调：虽然候选在验证集三种子比较中通过规则，但在封存测试集上，完整输入Macro-F1低于连续缺失基线的0.4857，MAE也高于基线的0.8045。因此，本次仅能确认验证集改进，不能宣称测试泛化提升。按预先固定的协议保留冻结模型，不利用测试结果改选。附件3无标签，不能报告其准确率。",
        "## 2 题目理解与数据接口",
        "问题2关注局部连续片段不可用时的预测稳定性，不要求重做问题1的原始特征提取，也不要求实现问题3的解释输出。缺失模态、缺失位置和缺失比例是需要控制的实验因素。三分类与连续强度是两个并行输出任务。",
        table(["用途", "样本数", "使用边界"], [["训练",3395,"拟合参数及标准化统计量"],["验证",728,"早停和模型选择"],["测试",727,"冻结后最终评价"],["附件3",30,"接口检查及最终预测，无标签"]]),
        "全部使用 aligned_50 版本。文本输入读取 text_bert 的词元编号通道，整数性、取值范围和注意力通道先行验证；不使用连续 text 字段，也不把两者作为不同模态。text_bert 在附件3中为 float32，但其值通过整数性检查。全量4850条编号、注意力掩码和分段编号与本地 bert-base-uncased 分词器逐项一致。轻量模型采用可训练嵌入，而非调用完整BERT编码器；因此没有借用已经看到完整文本的预编码表征。",
        "语音与视觉分别为50×74和50×35。类别0、1、2分别为Negative、Neutral、Positive，且与强度小于0、等于0、大于0逐条一致。三份官方划分没有重复样本ID。附件3以文件名和文件内从0开始的序号命名，例如 附件3_01.pkl::row000，不假造视频ID。",
        "## 3 有效位置与可观测掩码",
        "有效位置掩码回答某个位置是否属于样本内容；模态可观测掩码回答该位置的文本、语音或视觉是否可用。两者不能合并：填充位、CLS和SEP不是缺失的真实内容。有效范围由词元、注意力和其他模态的联合证据确定，排除第0位及可识别特殊词元，内部空缺仍保留为有效位置。",
        "原题将局部整条特征向量全零定义为缺失。因此在有效区间内，音频或视觉整向量为零才判不可用；单个分量为零仍是有效数值。若尾部所有模态同时为零且特殊词元也丢失，无法可靠区分缺失与填充，程序记录边界不确定性，不猜造长度。本批数据均能识别SEP边界。全部模态不可用时采用零汇总与有限数值输出，不能把这种兜底输出解释为可靠识别。",
        table(["划分", "内容位置", "语音零向量位置", "视觉零向量位置"], [[s,audit['audits'][s]['valid_positions'],audit['audits'][s]['modalities']['audio']['interior_all_zero'],audit['audits'][s]['modalities']['vision']['interior_all_zero']] for s in ['train','valid','test']]),
        "各模态标准化均值和标准差仅用训练集有效且可观测位置计算。标准差下限为0.00001，标准化后固定截断到[-10,10]，不可观测位置重置为零。验证、测试和专项样本均只使用保存的训练统计量。原文件不修改，哈希与处理后文件清单保存在审计记录中。",
        "## 4 模型与训练方法",
        "首版将词元映射为64维嵌入，语音和视觉分别线性投影到64维，经LayerNorm后按可观测位置进行掩码平均。三模态表示与三个可用比例拼接，经128维和64维前馈层，连接三分类头和强度回归头。回归输出为3乘以tanh，限制在[-3,3]。",
        "时序候选用三支小型双向GRU替代直接平均，每个方向隐藏维度32，并把位置可观测指示作为额外输入。打包序列避免尾部填充影响双向递归；缺失位置不参加最终汇总。门控候选用共享小网络读取各模态表示与可用比例，再归一化获得样本级权重；全不可用时权重和汇总均安全归零。",
        f"最终配置为 encoder={cfg['encoder']}，fusion={cfg['fusion']}，augmentation={cfg['augmentation']}，dropout={cfg['dropout']}，class_weight={cfg['class_weight']}，regression_weight={cfg['regression_weight']}。完整配置可直接加载，不依赖论文描述重新猜测。",
        "目标函数由三分类交叉熵和强度Huber损失加权相加，Huber转折点为1。类别加权候选采用训练集类别频次倒数归一化权重。AdamW学习率0.001、权重衰减0.01、初始批量64、最多40轮；梯度范数截断到5。每轮均用相同九种缺失验证场景选优，连续6轮未提高超过0.000001则早停。显存不足时批量减半，不终止其他程序。",
        "连续缺失训练默认对70%的样本随机选一种模态，按10%、30%或50%的有效位置数删除连续片段，其余样本保持原状。零散增强候选保持相同删除数量但分散采样位置。文本ID在嵌入和GRU之前置零，不能先编码完整语义再把结果置零。",
        "## 5 固定评价协议",
        "主验证共九种场景，即文本、语音、视觉分别删除10%、30%、50%的有效位置。每个样本与场景的随机起点由固定种子和样本ID的SHA256确定。数量四舍五入，非空样本至少删除1个位置；报告选中位置数与新增不可观测位置数，以区分原有缺失。补充场景采用前部、中部、后部和多个短段，三档比例均覆盖。没有可靠时间戳，因此仅报告位置数和比例，不换算为秒。",
        "内部选择分数为0.5乘以九场景平均Macro-F1，加上0.5乘以一减九场景平均MAE除以6。该式不是官方评分公式。候选在完整输入下，相对首版连续缺失模型的Macro-F1下降不能超过0.01，MAE增加不能超过0.03。阈值在首轮结果前已保存。",
        "最佳合格候选与基线在42、2026、3407三个相同种子下比较。至少两个种子同时提高内部得分并满足完整输入保护条件，且三种子均值也满足保护条件和得分提高，才接受候选。最终交付固定为选定结构的42种子模型，不从多个种子中挑最好的一次。均值±标准差反映本次随机初始化波动，不是置信区间。",
        "所有模型报告Accuracy、Macro-F1、各类F1、MAE和Pearson。F1固定包含三类，零分母记0。Pearson遇常量向量时未定义，记录原因而不填0。分类概率没有另行校准，不能视为真实正确率。",
        "## 6 对照与单因素比较",
        "多数类参考的验证Accuracy为0.4643、Macro-F1为0.2114；训练均值回归MAE为0.7808，中位数回归MAE为0.7690。常量回归的Pearson未定义。首版完整训练模型的MAE高于朴素参考，因此不能宣称两项任务都已明显改善。",
        table(["实验", "完整F1", "完整MAE", "缺失F1", "缺失MAE", "内部分数"], [[r['run'].replace('_s42',''),r['clean_f1'],r['clean_mae'],r['mean_missing_macro_f1'],r['mean_missing_mae'],r['score']] for r in summaries]),
        "候选按当前最佳合格配置每次只修改一个主要参数；因此表格是逐步消融路径，不是所有因素的全组合对照。每个实验的父配置和修改项保存在study/search.json，失败或不提升的记录均保留。",
        table(["候选", "父配置", "唯一修改", "刷新最佳"], [[x['run'].replace('_s42',''),x['parent'].replace('_s42',''),str(x['changed']),"是" if x.get('improved') else "否"] for x in search['candidates']]),
        f"本次停止新增候选的原因是 {search['stop_reason']}。预训练辅助为可选项，本轮没有执行；训练中未使用任何额外情感标注数据，也未开展未对齐版本搜索。",
        "## 7 三种子稳定性与最终选择",
        table(["模型", "完整F1", "完整MAE", "缺失F1", "缺失MAE", "内部分数"], [[role] + [f"{m[k]['mean']:.4f} ± {m[k]['std']:.4f}" for k in ['clean_f1','clean_mae','missing_f1','missing_mae','score']] for role,m in stability['summary'].items()]),
        table(["种子", "得分变化", "完整F1变化", "完整MAE变化", "通过"], [[r['seed'],r['score_delta'],r['clean_f1_delta'],r['clean_mae_delta'],"是" if r['passed'] else "否"] for r in stability['pairs']]),
        conclusion,
        "![训练过程中固定缺失验证分数](figures/learning_curve.png)",
        "## 8 冻结后评价与缺失规律",
        table(["划分和场景", "Accuracy", "Macro-F1", "MAE", "Pearson"], [[f'{s} 完整',e['clean']['accuracy'],e['clean']['macro_f1'],e['clean']['mae'],e['clean']['pearson']] for s,e in extended.items()]),
        table(["冻结后测试对照", "完整F1", "完整MAE", "缺失F1", "缺失MAE"], comparisons),
        "上述测试对照仅作冻结后外部检验，不据此重新选择模型。最终候选相对连续缺失基线的完整测试Macro-F1下降约3.04个百分点，MAE增加约0.0175；缺失场景平均F1同样下降。验证集选择偏差、划分之间的分布差异和结构归纳偏好均可能相关，但本轮实验不能区分其因果贡献。所有基线权重和预测保留于本地，供团队如实讨论这一负面结果。",
        table(["缺失场景", "验证F1", "验证MAE", "测试F1", "测试MAE"], [[s,valid[s]['macro_f1'],valid[s]['mae'],test[s]['macro_f1'],test[s]['mae']] for s in MAIN]),
        "![验证集缺失比例影响](figures/missing_impact.png)",
        f"在固定九种验证场景中，Macro-F1最低的是{worst_f1}，为{valid[worst_f1]['macro_f1']:.4f}；MAE最高的是{worst_mae}，为{valid[worst_mae]['mae']:.4f}。这是本次模型与固定掩码下的观测，不构成对所有缺失机制的排序。",
        "缺失导致的信息损失与噪声移除可能同时发生，单个场景的曲线不一定单调。应依据完整结果描述哪种模态、哪个比例更敏感，不能用少数起点宣称普遍规律。下表在相同30%预算下对比位置与片段形态，其余比例均保存在逐场景数据中。",
        table(["模态", "位置或形态", "验证F1", "验证MAE"], [[m,shape,valid[f'{m}_30_{shape}']['macro_f1'],valid[f'{m}_30_{shape}']['mae']] for m in ['text','audio','vision'] for shape in ['random','front','middle','back','multi']]),
        "## 9 错误分析与局限",
        table(["类别", "验证F1", "测试F1"], [[name,valid['clean']['class_f1'][i],test['clean']['class_f1'][i]] for i,name in enumerate(CLASSES)]),
        "![验证集混淆矩阵](figures/confusion_valid.png)",
        f"验证集两个输出头的类别与强度符号不一致共{disagreement}条，占{disagreement/728:.1%}。这包含中性分类而回归仅接近零的样本，不能简单等同于预测错误。当前不强制两个头一致，也不利用附件3的预测分布选择阈值。最大回归误差的10个样本及原文位于largest_valid_regression_errors.csv，便于逐例复核，但不能仅凭原文臆测音视频原因。",
        f"官方划分的相同视频ID交集数为{overlaps}。本实验保留官方划分；若存在同源视频片段跨划分，结果不应外推为严格跨视频或跨说话人泛化。掩码依赖原题全零定义，真实静默或检测器输出零可能存在语义歧义。人工随机删除不等价于真实传感器故障分布。",
        "模型规模小、随机词元嵌入语义先验弱，训练集只有3395条，复杂否定、长文本截断和模态冲突仍可能带来误判。本轮最多八个候选且仅三种子，不构成充分结构搜索或统计显著性证明。对完全不可用输入仅保证数值有限，不保证准确。",
        "## 10 附件3预测与字段说明",
        "专项CSV共30行。sample_id由文件名与行号构成；class_id取0/1/2；sentiment为对应英文类别；intensity范围[-3,3]；prob_negative、prob_neutral、prob_positive为三类softmax概率且和为1；scenario=clean表示不叠加人工缺失，原始缺失仍保留。表内为便于阅读四舍五入的展示值，提交CSV保留完整数值。",
        table(["文件", "类别", "强度", "负向概率", "中性概率", "正向概率"], [[r['sample_id'].split('::')[0],r['sentiment'],float(r['intensity']),float(r['prob_negative']),float(r['prob_neutral']),float(r['prob_positive'])] for r in special]),
        "## 11 复现和交付核验",
        f"运行环境为WSL2 Ubuntu 22.04，GPU {environment['gpu']}，Python 3.10，PyTorch {environment['packages']['torch']}，CUDA {environment['cuda']}。推理只需要模型、训练标准化参数及源码，不需要完整BERT权重。模型参数约200万，实际文件体积以交付清单为准。",
        "统一入口为python -m q2，包含audit、train、evaluate、predict和report。详细命令见README.md；固定配置、依赖版本、每轮日志、逐样本预测和场景掩码可核验。最终模型重载、单条与批量推理、CPU与GPU推理以及同种子重新训练均单独验证。",
        "最终核验发现cuDNN TF32使GRU单条与批量推理的强度相差约0.000038，超出原定0.00001容差。关闭推理TF32后差异降至约0.0000001；模型权重和验收阈值不变。训练保留原数值设置以验证精确重训。三种子选择表来自原训练评估，最终独立评价表来自统一FP32推理，因此小数尾位可能略有差异。",
        "正式参赛包与本地实验档案分开。正式包仅保留核心代码、最终参数、专项预测、方法报告及必要结果；原始数据和多次训练权重留在本地，不重复提交。全队仍需核算总附件不超过50MB，本题体积小于上限并不等于整队已满足。",
        "本报告表格来自保存的JSON和逐样本CSV，图形来源列于figure_sources.json。没有附件3真实标签，不提供其准确率；没有证明稳定提高时，以基线和负面实验结论交付。",
        "## 12 材料与方法依据",
        "题目依据为官方E题题面、附件2 aligned_50.pkl及附件3对齐版本。框架使用PyTorch、NumPy和scikit-learn，具体版本在requirements-lock.txt；本地bert-base-uncased分词器仅用于编号核验。所有性能数字以本项目实际日志为来源。"]
    return "\n\n".join(pieces) + "\n"
