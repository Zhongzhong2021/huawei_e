# 三问材料与运行

| 目录 | 内容 | 首先查看 |
|---|---|---|
| 问题1 | 全100条原生特征、时序关系、代码与配置 | `问题1/README.md`、`问题1/复现运行说明.md` |
| 问题2 | 定版模型代码、配置、验证结果、30条预测 | `问题2/results/attachment3_predictions.csv` |
| 问题3 | 预测解释代码、20条解释与对照结果 | `问题3/results/附件4_预测与解释汇总.csv`、`附件4_全20条解释卡.md` |

准备题目提供的原始数据。问题2和3统一使用对齐版本；三个问题分别安装运行环境。

## 下载与核验

百度网盘链接见`网盘下载说明.txt`。将对应ZIP解压到本目录：问题2得到`问题2/weights/model.pt`，问题3得到`问题3/weights/uncertainty/best.pt`。主模型推理需要这两个权重包。基线权重包用于重放SELF-MM和TETFN；画面包提供全20条解释卡引用的70张原尺寸画面。

```bash
python verify.py
# 四个外链包全部解压后：
python verify.py --external
```

`manifest.json`记录系统文件的SHA256，`external_files.json`记录网盘资源的SHA256。`网盘下载说明.txt`用于填写链接和提取码。问题1使用说明中指定的开源模型版本；问题2、3使用网盘中的训练权重。

## 问题1：读取特征与复现

`问题1/data/features`含100组NPZ及元信息，`data/multigranular_alignment`含对应关系；全部样本保留。使用`问题1/scripts/multigranular_reader.py`的`Dataset`接口读取。重新生成特征、下载工具模型及各环境的准确命令见`问题1/复现运行说明.md`。定位记录包括候选范围与支持状态；同期画面用于回看场景，发声身份记为未知。

## 问题2：专项推理

使用Python 3.10，进入`问题2`目录安装`requirements-retrain.txt`。CPU/GPU PyTorch按机器条件安装；原训练为PyTorch 2.8.0+cu128，完整版本见`environment.txt`。

```bash
cd 问题2
PYTHONPATH=src python -m q2v2.deploy \
  --checkpoint weights/model.pt --scaler weights/scaler.npz \
  --input-dir /绝对路径/附件3的对齐版本 \
  --output /绝对路径/新的附件3预测.csv --device cpu
```

采用种子42、4轮、连续缺失增强、逆频率分类权重与retain未知词口径。验证Accuracy/F1/MAE/Pearson为0.6195/0.6102/0.5861/0.6712；`results`中的46组指标与30条专项预测对应同一权重。

重训先按`scripts/prepare_bert_initialization.py --help`获取固定BERT初始化，再运行：

```bash
PYTHONPATH=src python scripts/retrain_seed42.py \
  --aligned /绝对路径/aligned_50.pkl \
  --pretrained /绝对路径/bert_base_uncased.pt \
  --output /绝对路径/新的训练目录
```

训练使用支持BF16的CUDA设备；完整优化参数见`configs/frozen_protocol.json`，实际训练4轮见`configs/retrain_seed42.json`。

## 问题3：专项预测与解释

在独立环境安装`问题3/requirements.txt`所列原训练依赖，或使用已验证的CPU重放环境：Python3.10、torch2.6.0+cpu、transformers4.48.3、numpy2.2.6（见`requirements-replay.txt`）。进入`问题3`：

```bash
HF_HUB_OFFLINE=1 python run.py predict --model uncertainty \
  --input /绝对路径/附件4的对齐版本 --split special \
  --output /绝对路径/新的问题3输出 --device cpu
```

输出`predictions.csv`和`explanations.jsonl`。预测和贡献在原特征索引上计算；随包结果将关键索引关联至文字、语音候选范围和同步画面，部分项保留定位支持不足状态。`tools/evidence_navigation`保留定位代码，固定模型记录与定位结果位于`results/evidence_navigation`，完整命令见`问题3/详细使用指南.md`。重新训练使用`python run.py train --help`，需另备原BERT预训练权重；三个模型的训练配置、归一化与决策策略均在`weights/<模型>/`中。

## 结果解释

验证集结果用于评价分类与强度预测性能。附件3和4为无标签专项集，CSV分别给出预测结果及解释。模型原理、数学表达、实验分析与全量结果表见论文PDF。
