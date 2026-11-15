# 植物幼苗识别：第二轮改进实验（2026-09-29）

## 已知结果和问题

用户确认上一版 ViT-L 的 `submission.csv` 在线得分约为 0.93，精确小数未知。原文件与 SHA-256 保存在 `baseline_vitl_submission.csv` 和 `baseline_online_report.json`。上一版内部嵌套交叉验证 0.944 与在线得分不是同一个指标来源。

数据为 500 张训练图（5 类各 100 张）和 378 张无标签测试图。像素哈希检查没有发现训练集完全重复图片；图像几乎均为正方形，原中心裁剪平均保留面积约 99.84%。因此本轮重点放在植物与背景的特征分离、方向变化和领域适配。

## 论文依据与实现边界

1. **Kumar et al., ICLR 2022, Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution.** 提出先线性探针、再微调（LP-FT），用于缓解随机分类头与骨干同时学习造成的表征改变。本轮将这一思路用于 ConvNeXt 和 DINOv2：每折仅在训练部分拟合逻辑回归头，转换为等价的 PyTorch Linear，再微调。DINOv2 只更新最后四层以适应 8 GB 显存。数据、架构与超参数不同，属于方法借鉴，不是原论文复现。
   - [论文](https://arxiv.org/abs/2202.10054)
   - [作者代码](https://github.com/AnanyaKumar/transfer_learning)
2. **Liu et al., CVPR 2022, A ConvNet for the 2020s.** ConvNeXt 提供具有卷积归纳偏置的现代骨干，用于与冻结 DINOv2 的全局表征形成不同的识别路径。本轮采用 torchvision 官方 ImageNet 预训练 ConvNeXt-Tiny；权重通过发布文件的 SHA-256 前缀核验。
   - [论文](https://arxiv.org/abs/2201.03545)
   - [官方仓库](https://github.com/facebookresearch/ConvNeXt)
3. **Pushpa et al., MethodsX 2026, SeedlingNet: A colour-based segmentation approach towards classification of plant species seedlings.** 论文摘要在植物幼苗数据上比较原图与颜色分割输入，支持检查背景干扰。本轮使用自己定义的固定 HSV/ExG 绿色掩膜给 DINOv2 patch 加权；这不等于复现其分割算法，也不把颜色掩膜当作人工分割真值。
   - [出版记录和摘要](https://pubmed.ncbi.nlm.nih.gov/41969976/)
   - [DOI](https://doi.org/10.1016/j.mex.2026.103883)
4. **Darcet et al., ICLR 2024, Vision Transformers Need Registers.** 论文分析了 ViT 内部特征图的异常 token，提示 patch 表征的空间汇聚需要检查。此次保留原有无 registers 的 ViT-L 权重，增加前景汇聚进行消融；没有宣称已经应用 registers 模型。
   - [论文](https://arxiv.org/abs/2309.16588)
5. **Siméoni et al., 2025, DINOv3.** 新一代自监督模型重点改善稠密空间特征，值得作为后续骨干候选。官方权重需要申请访问，本轮没有获得或使用这些权重；当前完成的模型仍是 DINOv2。不能把检索过 DINOv3 写成已经训练了 DINOv3。
   - [论文](https://arxiv.org/abs/2508.10104)
   - [官方代码与权重获取说明](https://github.com/facebookresearch/dinov3)

## 实验 A：方向增强和植物区域汇聚

保持原 ViT-L/14 权重和 518 输入。每图提取 0/90/180/270 度及各自水平翻转，共 8 个视角；缓存最后 4 层 CLS、末层全图 patch 均值和绿色区域加权 patch 均值。每张图的所有视角始终属于同一交叉验证折。

内层 4 折选择 2 或 8 视角、全图/前景/二者拼接、逻辑回归 C∈{0.01,0.1,1,10}；外层 5 折（seed=42）记录逐图预测。缩放参数只由当折训练部分拟合。最终以全部训练集重新选择分类器并生成测试预测。颜色掩膜可能包含浅色碎石或遗漏非绿色组织，因此保留全图特征作为候选。

## 实验 B：LP-FT 思路的 ConvNeXt 领域微调

每个外层训练折先拟合标准化逻辑回归分类头，再将权重和偏置转换为等价 Linear；代码包含等价性断言。接着更新整个 ConvNeXt 骨干与分类头，使用 288 输入、随机裁剪、90 度旋转、水平翻转、轻量颜色扰动、低概率 MixUp、AdamW、余弦学习率和 EMA。

原计划每折训练固定 20 个 epoch，使用最后的 EMA 状态。实际完成前两折分别为 0.92、0.88，低于原方案相同折的 0.98、0.92，因此停止后续训练，保留日志和两个完整模型。本实验没有完整五折分数，也没有参与最终候选融合。第三折中途诊断不是最终结果。

## 实验 C：将八视角分别作为分类头训练样本

复用实验 A 的特征缓存，但不先平均特征：每张图的八个视角分别进入逻辑回归，预测时平均八个概率。交叉验证先按原图划分，再展开视角，避免同一图旋转版本出现在训练和验证两侧。正则参数 C 除以 8，维持与原图数量相匹配的正则强度。

内层四折选择全图/前景/联合特征和 C∈{0.1,1,10}。外层五折结果为 0.96、0.94、0.94、0.95、0.97，汇总为 0.952；最终全训练集选用前景特征、C=10。该数值不是公榜成绩。

## 实验 D：DINOv2-L 的 LP-FT 领域适配

每个训练折先拟合分类头，再更新最后四个 Transformer block、末端 LayerNorm 和分类头。前 20 层冻结。输入 336，最后四层 CLS、全图 patch 和软 ExG 前景 patch 拼接为 6144 维表征。软掩膜没有分割标注监督，不宣称精确识别植物边界。

配置：固定 15 epoch，batch=4，梯度累积=2，骨干学习率 1e-5，分类头学习率 1e-4，AdamW weight_decay=0.05，两轮 warmup 后余弦衰减，label smoothing=0.03，25% 概率 MixUp，随机裁剪/直角旋转/翻转/轻量颜色扰动。EMA 衰减系数 0.98；每折使用最后状态，外层验证分数不用于挑 epoch。这里的 EMA 是这项植物分类实验预先固定的训练策略。

固定 seed=42 五折。验证图只由没有见过该图标签的模型预测；无标签测试集由五折模型、每个模型八视角平均。初始线性头的标准化器与逻辑回归均只接触当前折训练标签。诊断分数与最终八视角分数分开记录。

实际第一折权重差值已经核验：block 0 和 19 的 qkv 权重与预训练权重完全相等，block 20、23 及末端 norm 已改变，记录于 `dino_lpft/backbone_update_audit.json`。这证明微调更新了指定骨干层。

## 代码和复现命令

从项目根目录执行，依赖沿用当前 Python/PyTorch 环境，缓存目录为 `work/torch_cache`：

```powershell
python outputs/plant_v3_features.py extract
python outputs/plant_v3_features.py evaluate
python outputs/plant_v3_multiview.py
python outputs/plant_kernel_probe.py
python outputs/plant_dino_lpft.py
python outputs/select_plant_submission.py
```

保留同一配置重新执行微调，会复用已经完整保存的折；未完成折重新开始。修改配置应使用新的 `--run-dir`。推理脚本可从五个保存模型重新生成预测：

```powershell
python outputs/predict_plant_dino.py --output-dir outputs/stronger_20260929/dino_reprediction
```

推理脚本的 `--size` 可开展单独的分辨率实验，但本报告不能把未运行的命令写成已验证结果。

## 实验 E：RBF 分类边界对照

在相同冻结缓存上比较 RBF-SVM，内层四折选择特征汇聚方式和 C∈{0.1,1,10,100}，gamma 使用 scale。SVM 概率校准只接触外层训练子集。最终分类采用校准概率的 argmax，与提交推理一致；内层搜索采用 SVM 的常规分类准确率，两者可能略有不同。

五个外层最终准确率为 0.95、0.93、0.94、0.95、0.96，平均 0.946。这是一项常规非线性分类器对照，不是新的论文复现。所有完成的单模型与固定 50:50 的“冻结分类头 + 微调模型”概率平均均由 `select_plant_submission.py` 汇总比较，不拟合融合权重。

## 结果解释

本实验持续使用同一小训练集尝试新方法，因此跨实验选优仍可能让 CV 估计偏高。线上新成绩只有实际提交后才知道。每轮保存配置、数据哈希、OOF 概率、混淆矩阵和提交文件，避免将文献分数或内部验证分数写成公榜成绩。

完整结果见 [RESULTS.md](RESULTS.md) 和各实验目录的 `cv_report.json`；微调逐 epoch 日志为 `training.jsonl`。最终选择 DINOv2 LP-FT，五折 0.960，融合未提升。
