# 问题三：多模态情感识别与原生解释

本目录可独立搬移。主模型 `uncertainty` 将文本、音频、视觉证据进行不确定性加权融合；`selfmm`、`tetfn` 是两条开源方法的比赛数据适配基线。三类权重、运行代码和冻结结果随目录提供，原始比赛数据需另行放入 `data/`。推理无需联网，也无需单独下载 BERT 大权重：程序按随包配置建立结构，再严格加载任务权重。

## 快速使用

先阅读[环境配置](docs/环境配置.md)，在本目录安装 `requirements.txt`。建议把比赛给定的对齐文件放在 `data/aligned_50.pkl`，附件4的 `01.pkl` 至 `20.pkl` 放在 `data/attachment4_aligned/`。数据也可放在别处，命令用绝对路径指定。

```bash
cd /path/to/问题三
HF_HUB_OFFLINE=1 python run.py predict --model uncertainty --input "$PWD/data/attachment4_aligned" --split special --output "$PWD/output/uncertainty-special" --device cpu
```

主模型输出 `predictions.csv` 和 `explanations.jsonl`。两条基线只输出预测表。三个模型的完整命令、输入格式、训练、评价与排错见[详细使用指南](docs/详细使用指南.md)。权重经 Git LFS 分发，普通 `git clone` 后还需 `git lfs pull`；从完整 `tar.gz` 解包则不需要 LFS。运行 `python tools/package_delivery.py --verify` 可校验清单与权重完整性。

## 冻结模型结果

以下均为已冻结的**验证集 Macro-F1 目标**版本。完整精度、其他指标和专项样本数据见 [model_comparison.csv](results/model_comparison.csv) 及 `results/<模型>/metrics.json`。标签 `0/1/2` 分别为负向/中性/正向；MAE 越低越好。

| 模型 | 说明 | 轮次 | 验证 Accuracy / Macro-F1 / MAE | 测试 Accuracy / Macro-F1 / MAE |
|---|---|---:|---|---|
| uncertainty | 主模型，原生解释与不确定性融合 | 5 | 0.6511 / 0.6323 / 0.5397 | 0.6864 / 0.6493 / 0.6006 |
| selfmm | SELF-MM 比赛适配基线 | 3 | 0.6456 / 0.6390 / 0.5491 | 0.6616 / 0.6389 / 0.5966 |
| tetfn | MMSA TETFN 网络的比赛适配基线 | 1 | 0.6401 / 0.6321 / 0.5730 | 0.6561 / 0.6285 / 0.6226 |

附件4全部20条的正式展示见 [预测与解释CSV](results/附件4_预测与解释汇总.csv)和[全20条解释卡](results/附件4_全20条解释卡.md)，仅包含预测、模态作用与证据定位。论文为 [Word 文档](paper/问题三_不确定性融合论文.docx)，已按题目要求及三篇公开数学建模论文的写作结构修订；参考来源与修订对应关系见[论文写作参考与修订说明](docs/论文写作参考与修订说明.md)。安装`requirements-report.txt`后运行同目录 `build_paper.py` 可从冻结结果再生文稿、中文图表和正式结果。模型方法、代码来源和复现边界见[模块说明](docs/模块说明.md)与[模型来源与复现实验说明](docs/模型来源与复现实验说明.md)。

论文另补充原生结构四格对照和两组学习率试验，含较弱配置及相应参考配置。完整22组参数与来源见[补充对照实验说明](docs/补充对照实验说明.md)。

三份模型权重合计约 **1.3 GB**，超过比赛三题合计 **50 MB** 的附件上限；此目录是完整复现交付，不能直接作为符合该大小限制的比赛附件提交。
