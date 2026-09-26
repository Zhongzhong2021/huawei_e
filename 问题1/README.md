# 问题一：读取与复现

**GitHub入口：**首次在其他设备使用，请先阅读[跨设备继续工作](跨设备继续工作.md)。现有特征和必要代码已展开在本目录，无需另下载历史ZIP。

先读《问题一_正文.pdf》；同名Markdown可用于编辑，HTML便于浏览。本包保留一套主特征，不需要运行候选模型或额外扩展才能使用。

## 直接读取

Python 3.11、NumPy 2.2.6即可。在解压目录运行：

```text
python scripts/q1v6_reader_check.py --features features
python scripts/q1_final_audit.py --features features --output interface_check_new
python external_validation/verify_external.py features/samples
```

第一项期望100条、1948词、7905网格位置，来源摘要和填充检查通过。第三项使用附带公开记录复算统计，无需重新下载或GPU。

```python
import sys
sys.path.insert(0, 'scripts')
from q1v6_features import load_sample, collate
word, meta = load_sample('features', '-3g5yACwYnA__13', view='word')
grid, _ = load_sample('features', '-3g5yACwYnA__13', view='clock', step=0.1)
native, _ = load_sample('features', '-3g5yACwYnA__13', view='native')
batch = collate([word])
```

word_audio_dim_mask逐列区分音高缺失和其他声音特征；expression和scene独立标记。word中的场景掩码为word_scene_mask，grid中为scene_mask。duration_s是实际观测终点，container_declared_duration_s是容器声明时长；不能用后者补造有效数据。

## 从原始视频重建主特征

另用一份解压目录，results开始时为空，保留附带features作为对照。准备Docker及NVIDIA容器GPU支持；将.env.example复制为.env，设置DATA_ROOT_HOST为原题附件父目录，其中应有“附件1-数据集原始多模态样本”目录。模型和镜像首次下载需要网络，缓存保存在cache。

依次构建：

```text
docker compose build
docker compose -f compose.yaml -f compose.q1.yaml build
docker compose -f compose.yaml -f compose.q1-next.yaml build
docker compose -f compose.yaml -f compose.q1-next.yaml -f compose.q1-refinement.yaml build
```

随后按下列顺序，把程序名代入同一命令逐项运行；任一步失败须停止并查看日志，不能跳过：

```text
docker compose -f compose.yaml -f compose.q1-next.yaml -f compose.q1-refinement.yaml run --rm lab python scripts/程序名.py
```

1. q1_prepare、q1_extract、q1_quality、q1_scene
2. q1n_prepare、q1n_asr、q1n_posterior、q1n_audit
3. q1f_prepare、q1f_faces
4. q1v6_alignment、q1v6_media_clock、q1v6_features
5. q1v6_validate、q1v6_geometry、q1v6_visualize

输出位于results/q1_deep_v6/features_v2；运行读取检查时将--features指向该目录。生成程序拒绝覆盖已有最终目录。此流程中的对齐验证包含800次扰动推理；部署只读取特征时无需重跑。

reproduction/rebuild_protocol.json和rebuild_run.json记录已完成的空结果目录重建，3400个数组及掩码逐元素一致，成功步骤累计696.10秒。本次只整理正文与附件并重新核验，不重复声称完成一次新的全量GPU运行。硬件差异可能产生细小浮点差；应同时核对模型摘要、数值误差和掩码。Docker基础镜像及Python依赖固定，系统包仓库随时间可能变化，不能承诺未来构建的每个镜像字节完全相同。

## 证据和接口层次

features/protocol.json、input_contract.json定义最终接口；evidence下保留生成链各阶段原始参数及模型摘要，其中早期q1/run_config.json的声学汇聚是中间层定义，最终采用有声F0与97维输出。不得用早期配置覆盖最终定义。

external_validation提供同源公开结果和复算入口；其中文本对照实际使用附件二相同BERT上下文的预计算结果。公开参考不回填主输出，不作为人工真值。

method_choice仅保存候选方法选择的摘要和协议记录，完整原始候选输出留在工程历史研究档案，不属于本包主处理依赖。eGeMAPS、额外54维趋势及多模型候选向量均未拼入主特征。

本问自建词单位与附件二、三的预计算接口不同，不能因为同为768维就逐行替换。特征有效性也不等于情感预测效果。

## 文件完整性

manifest.sha256.json列出包内文件的SHA-256（不含清单自身和ZIP）。精简核验.json记录本次实际检查；所有100条主NPZ与上版完全相同。科研计算过程中的AI辅助来源见AI辅助使用记录与队内审阅要求.md。
