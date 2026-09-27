# 总论文

[阅读论文 PDF](paper.pdf)。论文依次说明数据与评价约定、三问的模型与求解、实验结果、模型评价和结论，正文包含附件一100条特征结果、附件三30条预测、附件四20条预测与关键证据索引。

## 编辑与编译

从仓库根目录运行：

```bash
make -C Paper
```

依赖 Python 3、Pandoc 2.9、XeLaTeX、latexmk、中文字体和 Poppler。Ubuntu 安装命令：

```bash
sudo apt-get install --no-install-recommends pandoc texlive-xetex texlive-lang-chinese texlive-fonts-recommended latexmk fonts-dejavu-core poppler-utils
```

构建读取既有实验结果和图表。保持 `Paper/` 与 `问题1/`、`问题2/`、`问题3/` 的相对位置。

| 编辑位置 | 内容 |
|---|---|
| `manuscript/question1.md` 至 `question3.md` | 三问正文源稿 |
| `introduction.tex`、`data.tex` | 问题分析、数据和评价约定 |
| `abstract.tex`、`conclusion.tex` | 摘要、模型评价和结论 |
| `main.tex`、`preamble.tex` | 文档结构、字体、图表与页码 |
| `scripts/sync_chapters.py` | 章节生成、表格列宽与分页、交叉引用 |
| `scripts/check_pdf.py` | 指标、结果表、来源及编译检查 |
| `assets/official-cover.pdf` | 由官方 Word 模板生成的封面 |

`chapters/` 由源稿自动生成。`make -C Paper sync` 同步章节，`make -C Paper check` 检查已生成的 PDF。构建结果写入 `build/verification.json`，图表来源摘要写入 `sources.json`。

封面源文件为 `references/official/template-2026.doc`。更新 Word 后，可运行 `python scripts/build_cover.py` 生成封面，再编译论文；Windows 使用 Microsoft Word，其他平台需要 LibreOffice Writer 和 Poppler。图内方法名的转换脚本为 `scripts/style_figures.py`，重新生成对应图形需要 CairoSVG，常规编译使用已有 PDF。

## 结果与版式

问题二采用 `q2-final-span-seed42-v1`，验证集 Accuracy、Macro-F1、MAE、Pearson 分别为0.6195、0.6102、0.5861、0.6712，来源为 `../问题2/docs/final/`。方案选择阶段的三种子结果单独报告。模型权重及标准化参数已核对SHA256，已有CPU重放的30条类别与固定结果全部一致。

中文正文与表格保持小四宋体样式，一级标题为四号黑体，题目为三号黑体。表格按内容分配列宽，数值右对齐，短表整体排版，长表重复表头；摘要起连续编号。Linux环境使用Fandol宋体／黑体和嵌入字体。

自动检查核对100条特征、30条预测、20条预测与证据索引、验证指标和引用，并检查缺字、版面越界及页码。页面视觉检查与本轮修改记录见 [修订验证](planning/修订验证.json)，题目要求映射见 [审查清单](planning/审查清单.csv)。

论文整理、润色和构建使用OpenAI Codex及Humanizer-zh辅助，具体范围见 [AI使用记录](planning/AI使用记录.md)。
