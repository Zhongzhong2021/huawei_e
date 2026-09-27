# 总论文

[阅读PDF](paper.pdf)，当前44页（含封面、1页摘要与参考文献）。2026-09-27参考用户提供的十篇2023年E题论文重新整理叙述结构：问题重述与分析、模型假设与数据准备、三问建模与求解、模型评价与改进方向、结论。正文保留附件一100条特征结果、附件三30条预测、附件四20条预测与解释；另根据已有CSV补充问题三后处理前后MAE对照。

## 编译与编辑

问题二已按用户补充的`paper.pdf`同步至`q2-final-span-seed42-v1`：验证集Accuracy/Macro-F1/MAE/Pearson为0.6195/0.6102/0.5861/0.6712，图表与附件三30条预测统一使用该版本。最新记录见`../问题2/docs/final/`；历史三个随机种子统计用于说明训练方案选择，与最终模型分开报告。来源快照见`references/q2-update/`。

本地Windows便携环境：在本次交付的`outputs/`目录运行`./rebuild_paper.ps1`，依次从已保存的Word生成封面、同步Markdown、编译、核对并生成`论文_结构调整版.pdf`。封面转换使用本机Microsoft Word，需保留同级`../work/tools/`中的Pandoc 2.9.2.1、Tectonic和字体。原本仅执行LaTeX编译的`compile_pdf.ps1`仍可使用，但不会同步Word封面或Markdown。`-SkipSync`会跳过这两项同步。

从仓库根目录运行：

```bash
make -C Paper
```

本机验证版本为Python 3.10、Pandoc 2.9.2.1和TeX Live 2021。依赖Python 3、Pandoc 2.9、XeLaTeX、latexmk、中文字体和Poppler。Ubuntu可安装：

```bash
sudo apt-get install --no-install-recommends pandoc texlive-xetex texlive-lang-chinese texlive-fonts-recommended latexmk fonts-dejavu-core poppler-utils
```

编译不调用模型，也不下载原始数据。保持`Paper/`与三个问题目录的相对位置，图表由现有结果读取。

| 编辑位置 | 内容 |
|---|---|
| `manuscript/question1.md` 至 `question3.md` | 经过取舍和编辑的正式章节源稿 |
| `introduction.tex`、`data.tex` | 总体路线、数据和共用评价约定 |
| `abstract.tex`、`conclusion.tex` | 摘要与结论 |
| `main.tex`、`preamble.tex` | 文档结构、字号、图表及页码 |
| `assets/official-cover.pdf` | 官方模板转换的封面，四个标识保留 |
| `chapters/` | 从正式源稿生成的LaTeX；不直接编辑 |
| `scripts/sync_chapters.py` | 渲染章节、处理原生交叉引用与首次引用顺序 |
| `scripts/check_pdf.py` | 核对分页、结果表、指标、引用、字形及来源摘要 |
| `sources.json` | 正式源稿、研究源稿、图表与模板输入的SHA256 |

`make sync`只更新生成章节，保留正式Markdown编辑；`make check`核对已生成PDF。原问题目录的研究稿作为来源保存，修改后需判断是否同步到正式章节。问题三公式由原实现生成器中的LaTeX定义读取。

首页编辑入口为`references/official/template-2026.doc`。保存Word后运行`outputs/rebuild_paper.ps1`，脚本会重新生成`assets/official-cover.pdf`并编入整篇论文；保存Word本身不会触发自动编译。转换只取第一页，并在转换副本中删除页脚，源Word保持不变。单独执行`python scripts/build_cover.py`也可生成封面：Windows使用Microsoft Word，其他平台需要LibreOffice Writer及Poppler。

## 内容与验证范围

[修订说明](论文规划.md)记录结构、取舍和表述规则，[审查清单](planning/审查清单.csv)逐项对应题目和官方规范，[参考资料](references/README.md)记录官方来源及获奖论文阅读依据。

自动核对包括：100条序列长度、时长与词状态对应原结果；30条预测对应既有研究稿；20条预测、模态占比、主要模态及定位帧对应CSV；问题三验证和测试指标对应JSON；摘要与正文页码连续；参考文献按首次引用排列；无未定义引用、缺字或版面越界。可视检查另外记录于`planning/修订验证.json`。这些检查验证文稿与已有结果的一致性，不代表重新训练模型或取得独立准确率真值。

当前环境使用Fandol宋体／黑体，中文正文与表格为小四，标题按三号／四号设置；全部字体嵌入PDF。官方Word封面转换时使用本机替代字体，未声称与Windows宋体、黑体逐字形一致。

问题二的权重和标准化参数未在本工作区取得，30条预测核对至既有记录，未作本轮独立模型重放。问题一、三的复现结论按既有记录陈述。

论文整理与构建代码使用OpenAI Codex（GPT-6）辅助。使用日期、范围与实际标注见[AI使用记录](planning/AI使用记录.md)；完整工具版本发布日期尚待核实。正式封面身份信息和该元信息须按实际参赛资料补齐。

图内方法名通过`scripts/style_figures.py`修改原SVG文本，数值与几何图形保持一致，来源记录于`assets/figure_sources.json`。仅重新生成这两幅图时需要CairoSVG 2.9.1，常规编译直接使用已生成PDF。
