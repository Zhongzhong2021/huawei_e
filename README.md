# huawei_e

## 提交材料

[submission：论文、系统附件与外链资源](submission/README.md)。系统附件约25.8MB，权重与解释画面放在“网盘资源”目录；上传百度网盘后填写下载说明。

## 整体论文

- [Paper：LaTeX总论文与构建说明](Paper/README.md)
- [整体论文PDF](Paper/paper.pdf)

在仓库根目录执行 `make -C Paper`，从Paper内的正式源稿、公式和结果图表生成PDF。

## 问题1

- [方法、结果与交付入口](问题1/README.md)

问题1基于附件1完成100条样本的三模态特征提取与时序对应，提供正文、全量特征、典型样本及复现材料。

## 问题2

- [代码与运行说明](问题2/README.md)
- [论文Word与Markdown文档入口](问题2/docs/README.md)
- [当前模型迭代效果](问题2/docs/当前迭代效果.md)

问题2已统一为[定版重训模型、验证图表与30条专项预测](问题2/docs/final/README.md)。原始数据另行准备，大型权重按submission清单通过外链提供。


## 问题三

- [完整交付与快速使用](问题3/README.md)
- [详细使用指南](问题3/docs/详细使用指南.md)与[环境配置](问题3/docs/环境配置.md)
- [论文Word](问题3/paper/问题三_不确定性融合论文.docx)与[可编辑正文](问题3/paper/问题三_不确定性融合论文.md)
- [附件4全部20条解释卡](问题3/results/附件4_全20条解释卡.md)与[交付验证报告](问题3/docs/交付验证报告.md)

主模型为不确定性融合；开源适配基线为SELF-MM和TETFN。源码、三个完整任务权重、冻结预测解释和文档随目录提供；权重约1.3GB，通过Git LFS分发，克隆后运行 `git lfs pull`。比赛对齐数据需自备。提交时按submission中的系统附件与外链资源分别提供。
