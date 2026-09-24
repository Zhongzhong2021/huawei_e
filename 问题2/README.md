# E题问题2 代码与运行说明

本目录为第三轮研究成果的代码快照，负责局部模态缺失条件下的情感三分类与强度回归。源码与已验收的第三轮交付版本保持一致；不在这次仓库迁移中重新训练或改变模型。

最终模型为完整预训练词表的12层文本编码器与音视频特征融合模型，使用连续缺失训练。三种子官方验证完整输入Macro-F1均值为0.5959、MAE均值为0.5783；第二轮FP32基准分别为0.5409、0.6534。本轮消融未证明缺失增强本身带来稳定收益，不能将整体模型改善归因于这一因素。测试集仅作冻结后的描述性评价。

## 目录

- `src/q2/`：原始数据接口、掩码、轻量基线、指标与通用训练推理。
- `src/q2v2/`：预训练文本模型、训练器、蒸馏及离线部署；量化模块仅用于兼容历史模型，第三轮最终模型未量化。
- `src/q2v3/`：统一入口、分组统计分析和科研绘图。
- `scripts/`：原研究流程、冻结评价、论文生成与验收脚本。
- `tests/`：18项自动测试，覆盖数据、标签、掩码、缺失信息隔离、空输入、词表及Bootstrap配对等。
- `configs/confirmed_seed42.json`：已确认模型的原始配置。`max_epochs=6`是内部开发上限；最终确认与固定配方重训实际使用4轮，轮数保存在冻结模型检查点内。
- `metadata/frozen_model.json`：原模型冻结记录及SHA256。原运行绝对路径仅作历史追溯，不要求队友使用相同路径。

## 数据与模型分开共享

Git仅保存代码、测试、配置与说明，不提交赛题原始数据、大型权重、逐样本实验档案或压缩包。

配套研究资料来自团队共享的 `E题问题2_第三轮进展_队友交接包_20260924.zip`。其解压结构为 `E_Q2_round3_team_handoff/round3/`，包含最终模型、第二轮FP32对照、真实实验记录、Word与Markdown论文素材、16个公式、8组科研图及30条附件3预测。

最终模型文件校验值：

```text
final/model.pt
SHA256 2b91c28f9d3087f61438443ad9f0a365c136406f34314d4792f8b2cf60cdd407

final/scaler.npz
SHA256 30bfe7e4d67527107f6931794e53d5a5f977517dd644ac477cb5ca3b38285d6c
```

可以把压缩包中的 `round3/final/` 复制到本目录的 `final/` 下；该目录已被Git忽略。也可不复制，使用下面的 `--root` 直接指向交接包解压目录。

## 环境

已验证环境为WSL2 Ubuntu 22.04、Python 3.10.18、PyTorch 2.8.0+CUDA 12.8、NumPy 2.2.6、scikit-learn 1.7.2、SciPy 1.15.3、Matplotlib 3.10.6，依赖见 `requirements-lock.txt`。

优先复用已验证的 `e_q2` 环境。新机器应先按GPU与CUDA环境安装相容的PyTorch构建，再安装其余锁定依赖；不要为运行本代码修改显卡驱动。仅查看论文与图形无需安装训练环境。

## 先运行测试

在WSL终端中进入本仓库的“问题2”目录：

```bash
export PYTHONPATH="$PWD/src"
python -m q2v3 audit
```

这些自动测试不需要赛题数据或模型权重。仅在当前目录成功执行 `audit`，不等同于完成真实模型推理核验。

## 附件3离线预测

如果已将交接包的 `final/` 复制到本目录：

```bash
python -m q2v3 predict \
  --input-dir /你的路径/附件3对齐版本 \
  --output predictions.csv \
  --device cpu
```

如果要直接使用解压包中的模型：

```bash
python -m q2v3 --root /你的路径/E_Q2_round3_team_handoff/round3 predict \
  --input-dir /你的路径/附件3对齐版本 \
  --output predictions.csv \
  --device cpu
```

可用GPU时把 `cpu` 替换为 `cuda`。推理使用本地冻结模型与标准化参数，不联网下载，也不依赖教师或原始BERT初始化权重。输入必须是赛题对齐版PKL接口，不可临时换用未对齐版；不要加载来源不可信的PKL。

输出字段为 `sample_id, scenario, class_id, sentiment, intensity, prob_negative, prob_neutral, prob_positive`。类别0负向、1中性、2正向，强度范围为[-3,3]。程序同时检查单条与批量预测、重新加载、样本标识唯一性、有限数值和概率和，并保存 `.checks.json`。

附件3没有真实标签，全部30条预测覆盖不代表已知准确率。稳定标识由文件名与文件内序号组成，不冒充原视频ID。

## 使用交接包复算统计与图形

下面的 `--root` 必须指向完整交接包中的 `round3`，而非仅有代码的仓库目录：

```bash
python -m q2v3 --root /你的路径/E_Q2_round3_team_handoff/round3 analyze
python -m q2v3 --root /你的路径/E_Q2_round3_team_handoff/round3 figures
```

`analyze`从保存的逐样本预测复算184组指标和2000次按视频分组的配对Bootstrap；`figures`生成8组SVG、PDF和600 dpi PNG。不要手改图表数值。

`paper`生成Markdown和结构化正文；生成原生可编辑公式Word另需python-docx、lxml以及 `scripts/build_paper.py` 指定的Office `MML2OMML.XSL`。在没有python-docx的环境中仅生成Markdown和结构化数据，不生成Word。直接使用交接包中的已核验Word最方便。

## 固定配方重训

重训需要另行准备官方数据和本地预训练初始化权重，二者不在Git中，也不等同于最终推理模型。先按 `q2.data.prepare` 的规则整理数据，保证训练统计量只由训练集估计；必要时使用原始接口：

```bash
python -m q2 --root /你的路径/数据工程 audit \
  --source /你的路径/附件2对齐版.pkl \
  --special /你的路径/附件3对齐版本
```

在本目录准备好 `final/model.pt` 后，使用新的实验名称：

```bash
python -m q2v3 train --name reproduction_new --seed 42 \
  --data-root /你的路径/数据工程 \
  --pretrained /你的路径/bert_base_uncased.pt
```

此命令按冻结结构、完整词表和4轮固定配方重新训练，拒绝覆盖已存在的同名实验。它不启动新的模型选择。若需要真正开展新一轮优化，应建立新协议与新实验目录，不改写第三轮冻结证据。

## 历史脚本的边界

`run_study.py`保留原绝对截止时间、原第二轮依赖及冻结保护，仅作为研究过程记录；不要直接在新机器上用它无限重复搜索。

`evaluate_frozen.py`、`audit_release.py`和`verify_delivery.py`是原完整研究目录的验收工具，分别可能需要原预处理数据、重复训练检查点、旧版档案或人工逐页审阅后的渲染文件。不能把缺失这些资源造成的错误解释为模型精度不达标。队友优先使用上面的便携 `audit` 与 `predict` 命令。

所有逐样本结果、区间、图表源数据和验收详情以交接包为准。正式提交前仍须统一整队论文格式、匿名要求和附件大小。
