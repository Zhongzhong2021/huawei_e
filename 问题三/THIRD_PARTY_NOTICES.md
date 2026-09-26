# 第三方来源、许可和本交付修改

本目录主模型是本地设计的上下文化加性读出与不确定性融合模型。两个基线为开源架构的比赛适配；不声称论文原设置复现或上游官方发布的比赛权重。

| 内容 | 上游与固定版本 | 复用方式与本包修改 | 许可 |
|---|---|---|---|
| BERT及其衍生训练权重 | google-bert/bert-base-uncased，86b5e0934494bd15c9632b12f734a8a67f723594；Transformers 5.15.1 | 编码器12层微调；本包推理从config初始化后载入完整任务checkpoint，不再下载基准权重 | Apache-2.0，licenses/BERT-Apache-2.0.txt |
| SELF-MM适配 | thuiar/MMSA，a94e65d07fa1ae0d44e552390074b29b0898edfd，models/multiTask/SELF_MM.py 与 trains/multiTask/SELF_MM.py | q3b/selfmm_adapter.py 适配网络、中心及伪标签训练；增加三分类输出、观测打包与数值稳定；未逐字复制完整训练器 | MIT，licenses/MMSA-MIT.txt |
| TETFN源码适配 | 同版MMSA，models/multiTask/TETFN.py及实际subNets | third_party/MMSA保留TETFN实际依赖子树；BertTextEncoder.py修改为本地config/权重加载并跳过未使用的tokenizer；其余保留算法。q3b/tetfn_source_adapter.py恢复对齐音视位置、增加分类头和有界输出，本地训练器取代原动态标签训练器 | MIT，third_party/MMSA/LICENSE 与 licenses/MMSA-MIT.txt |
| 数据与训练基础流程 | 同版MMSA data_loader.py和trains/singleTask/EF_LSTM.py | split-pickle组织与常规训练思路适配；本地观测掩码、训练统计、多目标验证选择及解释 | MIT，licenses/MMSA-MIT.txt |
| 低秩乘积运算 | pliang279/MultiBench，49f9be224f342de005b7f001281f33df4301eb22，fusions/common_fusions.py | q3b/model.py参考投影和逐元素乘积，设计无偏置成对读出；没有复制完整LRTF类 | MIT，licenses/MultiBench-MIT.txt |
| 可加网络思想 | google-research/google-research，d36068b845da4c2b24927fee2cea1e6ef98dadda，neural_additive_models | 研究参考，没有引入TensorFlow实现或把NAM作为已训练基线 | 上游Apache-2.0，无代码分发 |
| 媒体定位结果 | torchaudio.pipelines.MMS_FA，torchaudio 2.5.1+cu124 | results/media_mapping.jsonl为历史MMS强制对齐与词元核验结果，仅复用数据定位，非情感预测；本包不含MMS权重，不在主模型运行时调用 | MMS对齐权重上游CC-BY-NC 4.0，https://github.com/facebookresearch/fairseq/tree/100cd91db19bb27277a06a25eb4154c805b10189/examples/mms#license |

源仓库：https://github.com/thuiar/MMSA 、https://github.com/pliang279/MultiBench 、https://github.com/google-research/google-research 。权重的SHA256及来源轮次见weights/*/selection.json与manifest.json。比赛原始特征、视频及完整公开数据未随交付分发。
