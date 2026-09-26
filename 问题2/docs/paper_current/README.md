# 问题2统一论文素材的生成与核验

当前稿以第四轮冻结的 `balanced_inverse` 为正式模型，以第二、三轮为历史对照，并纳入第五、六轮负面结果。它是论文整合版本，不是第七轮模型。原轮次文档与模型均未覆盖。

## 文件入口

- [正文Markdown](reports/E题问题2统一论文素材与当前模型结果.md)与[可编辑Word](reports/E题问题2统一论文素材与当前模型结果.docx)：20页、19个原生公式、18张表、2个算法表、9组图。
- [LaTeX公式源码](reports/current_equations.tex)及[结构化正文](reports/current_content.json)：与Word、Markdown同源。
- [当前模型证据](analysis/current_evidence.json)：46类场景的两模型指标、当前案例、附件3全部预测、三种子确认及真实训练日志摘要。
- [内容来源](reports/current_provenance.json)：逐块原轮次、编号映射、生成脚本与复制图形哈希。
- [预测复算记录](evidence/evidence_checks.json)、[论文表格核验](evidence/manuscript_checks.json)、[20页排版核验](evidence/document_qa.json)：三项检查分别证明数值来源、稿件一致性与可读性，不证明泛化性能。

图1、图2保留第三轮的相同结构和历史消融；图3至图5、图7来自第四轮已核验图；图6、图8、图9由第四轮冻结预测和内部折日志重新生成。所有图均提供SVG、PDF及600 dpi PNG，预览图和分页QA图不进入交付目录。

## 复现流程

数值与图形生成入口为 `scripts/prepare_current_paper.py`，需提供完整第四轮导出目录、第四轮训练日志目录和一个不存在的新输出目录：

```bash
python scripts/prepare_current_paper.py --source /完整实验档案/round4 --runs /完整实验档案/round4/runs --output /新的论文工作目录
```

该步骤使用NumPy和Matplotlib，复算92个验证预测CSV的指标并核对配对、案例和附件3，不训练、不选择模型、不重新评价测试集。本机使用既有WSL `e_q2` 科研环境。原始CSV和模型权重不在此文档目录内，团队共享完整实验档案后才能从头复算。

Word与Markdown生成入口为 `scripts/paper_current.py`，`--docs` 指向本仓库 `问题2/docs`，`--root` 指向上一步输出。Word生成使用已解析的文档Python运行时、python-docx、lxml和本机Office的MML2OMML样式表；无Office环境可使用已交付的Word和LaTeX源码，不保证该特定构建器直接跨平台运行。

`scripts/audit_current_paper.py --root <论文工作目录> --docs <docs目录>` 会核对来源哈希、当前数值表、历史数据不变性及连续编号。来源记录包含生成机绝对路径；其他设备应按路径映射恢复原始档案，不能删除来源校验来冒充复现。

最后渲染Word并逐页检查，再运行 `scripts/verify_paper_material.py <论文工作目录> <QA目录> --pdf-name current.pdf --page-padding 2 --reviewed-pages 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20` 记录本次审阅。页数随环境或改稿变化时必须实际复查后调整，不能直接沿用旧验收。重新排版后的文件哈希不同即需重新验收。

## 论文使用边界

三种子均值描述方案稳定性，附件3使用固定种子42的单一冻结模型。内部交叉验证与官方验证已参与多轮开发，测试集结果是冻结后的描述性结果，不是独立盲测。最大类别概率未校准；附件3没有提供标签，不报告其准确率。表和图中的非单调、不利变化及区间跨零情况必须保留。
