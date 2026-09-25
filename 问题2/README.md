# E题问题2 代码与运行说明

本目录负责局部模态缺失条件下的情感三分类与强度回归。2026年9月25日完成第四轮类别平衡实验，并补充便携重训、只读预检查、独立冻结评价和按研究轮次自动选择的图文入口。第三轮模型、结果与原始交接压缩包保持不变。

当前冻结模型为完整预训练词表的12层文本编码器与音视频融合模型，保留连续缺失训练，新增仅由拟合样本估计的逆频率分类权重。三种子官方验证完整输入Macro-F1由第三轮的0.5959升至0.6163，中性F1由0.3871升至0.4808；MAE由0.5783变为0.5790。三个同种子配对均通过预设保护条件。

必须同时报告权衡：平均Accuracy和正向F1下降，固定种子整体Macro-F1的分组Bootstrap区间跨零；官方验证已经多轮复用，不宣称无偏泛化提升或竞赛排名。第三轮消融也未证明缺失增强本身带来稳定收益。测试集只作冻结后的描述性评价。

## 目录

- `docs/`：[论文文档与当前进展](docs/README.md)，包含第一至四轮Word与Markdown、第三及第四轮科研图、公式源码、表格证据，以及第五轮未完成探索的状态说明。
- `src/q2/`：原始数据接口、掩码、轻量基线、指标与通用训练推理。
- `src/q2v2/`：预训练文本模型、训练器、蒸馏及离线部署；量化模块仅用于兼容历史模型，第三轮最终模型未量化。
- `src/q2v3/`：统一入口、分组统计分析和科研绘图。
- `scripts/`：原研究流程、冻结评价、论文生成与验收脚本。
- `tests/`：35项自动测试，覆盖数据、标签、掩码、缺失信息隔离、空输入、词表、Bootstrap配对、加权梯度累积等价性、重训预检查与新旧轮次入口路由。
- `configs/confirmed_seed42.json`：已确认模型的原始配置。`max_epochs=6`是内部开发上限；最终确认与固定配方重训实际使用4轮，轮数保存在冻结模型检查点内。
- `configs/frozen_protocol.json`：第四轮预定协议完整副本，提供AdamW学习率、批量、权重衰减与选择边界；不依赖被Git忽略的 `study/` 目录。
- `metadata/frozen_model.json`：第四轮冻结记录及SHA256。原运行绝对路径仅作历史追溯，不要求队友使用相同路径。
- `configs/round3_*.json`、`metadata/round3_frozen_model.json`：第三轮配置与冻结信息，不与第四轮模型混用。

## 数据与模型分开共享

Git保存代码、测试、配置、论文文档、科研图和必要汇总证据，不提交赛题原始数据、大型权重、完整逐样本实验档案或压缩包。[文档阅读入口](docs/README.md)和[当前模型迭代效果](docs/当前迭代效果.md)已经随仓库同步；只读文档无需下载模型。

配套研究资料来自团队共享的 `E题问题2_第三轮进展_队友交接包_20260924.zip`。其解压结构为 `E_Q2_round3_team_handoff/round3/`，包含最终模型、第二轮FP32对照、真实实验记录、Word与Markdown论文素材、16个公式、8组科研图及30条附件3预测。

第四轮完整材料独立保存在当前Windows工作区的 `output/e_q2/round4/`。其中 `reports/` 是8页论文补充Word、Markdown、4个可编辑公式与8张数据表；`figures/` 是4组SVG、PDF和600 dpi PNG；这些文档与图已同步至仓库 `docs/round4/`。`final/` 是另行共享的当前模型、标准化参数及30条附件3预测。旧第三轮交接包不包含第四轮模型。

当前第四轮模型文件校验值：

```text
final/model.pt
SHA256 56c8e2267dfa31f29929af9c68940d169ebe7fac3f1020cd05d697db0ea86e2e

final/scaler.npz
SHA256 30bfe7e4d67527107f6931794e53d5a5f977517dd644ac477cb5ca3b38285d6c
```

要使用新模型，请把第四轮的 `final/` 复制到本目录的 `final/` 下；该目录已被Git忽略。仅有第三轮交接包时，直接用 `--root` 指向其 `round3/`，保留其中的历史协议，不将第三轮模型放进第四轮元数据目录后混用。

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

第四轮使用下列统一入口；必须指向包含完整证据的研究目录，不能只指向源码仓库。程序读取该目录的协议，自动选择正确轮次的图表与论文生成器。

```bash
python -m q2v3 --root /你的路径/round4 analyze
python -m q2v3 --root /你的路径/round4 figures
python -m q2v3 --root /你的路径/round4 paper
```

第四轮`analyze`包括2000次视频分组配对Bootstrap与全部类别的权衡诊断；`figures`输出4组新增图。`paper`在WSL无Word依赖时生成Markdown和结构化表格；Windows原生可编辑公式Word需要python-docx、lxml与Office数学样式表。交付Word已经过逐页检查。第三轮完整方法正文和第四轮补充应共同使用，不将旧模型成绩误写为当前模型成绩。

重新计算冻结模型的46种验证场景，需要官方预处理数据和训练集标准化参数：

```bash
python -m q2v3 --root /你的路径/round4 evaluate \
  --data-root /你的路径/数据工程 --device cuda \
  --split valid --output-dir /你的路径/独立复算结果
```

输出目录必须尚不存在；模型和标准化参数不匹配时明确失败，不覆盖已有预测。`--split test`只用于冻结后的描述性测试。推理数据必须由该模型相同的训练集标准化参数生成。

以下保留第三轮历史复算示例：

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

在本目录准备好 `final/model.pt` 后，先使用新的实验名称执行只读预检查：

```bash
python -m q2v3 train --name reproduction_new --seed 42 \
  --data-root /你的路径/数据工程 \
  --pretrained /你的路径/bert_base_uncased.pt \
  --dry-run
```

`--dry-run`检查必要文件、协议参数与冻结轮数，不分配训练模型、不拟合数据，也不创建实验目录。它不代替数据内容审计或真实训练验证；缺文件时会在加载权重之前列出清单。检查通过后，移除 `--dry-run` 才会执行重训。可用 `--checkpoint /路径/model.pt`直接引用外部冻结模型，无需复制大文件。

协议按 `--protocol`显式指定、`study/protocol.json`、`configs/frozen_protocol.json`的顺序选择。完整研究目录继续使用原协议，代码仓库使用已跟踪的冻结副本。实际训练会把完整协议另存为 `runs/<name>/training_protocol.json`，避免只记录网络配置却漏掉优化器参数。

真正的重训命令按检查点中的结构、类别权重配置、完整词表和4轮固定配方执行，拒绝覆盖已有同名实验。它不启动新的模型选择。第四轮已在同环境重训，217个参数张量完全相同，检查点哈希一致；不承诺跨硬件逐位一致。若需要新一轮优化，应先确认预算并建立新协议与新目录，不改写任何已冻结证据。

## 历史脚本的边界

`run_study.py`保留原绝对截止时间、原第二轮依赖及冻结保护，仅作为研究过程记录；不要直接在新机器上用它无限重复搜索。

`run_round4.py`是第四轮预注册的两配置研究流程；`finish_round4.py`只用于一次性冻结后评价；`audit_round4.py`独立核对选择规则、拟合类别计数和参数复现。完整中间检查点留在WSL研究档案，Windows交付目录仅保留最终权重。研究期源码摘要见`audit/source_hashes.json`；交付时新增的便携命令属于后续工具完善，不宣称与研究期所有源文件逐字相同。

`evaluate_frozen.py`、`audit_release.py`和`verify_delivery.py`是原完整研究目录的验收工具，分别可能需要原预处理数据、重复训练检查点、旧版档案或人工逐页审阅后的渲染文件。不能把缺失这些资源造成的错误解释为模型精度不达标。队友优先使用上面的便携 `audit` 与 `predict` 命令。

所有逐样本结果、区间、图表源数据和验收详情以交接包为准。正式提交前仍须统一整队论文格式、匿名要求和附件大小。
