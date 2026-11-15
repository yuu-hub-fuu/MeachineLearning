# 植物分类 v4：从已报告的 0.9735 继续改进

## 基线与评价

用户在收到 `stronger_20260929/submission_v3_20260929.csv` 后报告线上分数 **0.9735**。按紧接的提交反馈关联该文件，SHA-256 为 `88e0830d1743da1cf3f528e6437525b886e903da9905a39fe24a9e53a040a882`；未独立登录平台查分。该模型的内部五折准确率为 0.960，二者来自不同数据，不相互替代。

本轮仍使用同样的 500 张训练图、378 张无标签测试图及 seed=42 的分层五折。每张图的所有增强视角留在同一折。分类头的标准化和超参数选择只接触外层训练子集。跨实验复用开发折仍会带来选优偏差，新线上结果需要实际提交。

## 1. 分辨率对照

使用上一版五个已保存 DINOv2-L 微调模型，将评估输入从 336 改为 518；每图仍平均八个旋转/翻转视角。没有重新训练，也没有使用测试标签。保存每折验证和测试概率，随后比较原分辨率、高分辨率以及预先规定的等权多尺度平均。

代码：`outputs/plant_v4_resolution.py`。完整结果写入 `dino_518/cv_report.json`。

## 2. 植物区域的尺度归一化

对原图使用固定 HSV/ExG 条件检测绿色区域，去除极小连通区域，以绿色像素 1%–99% 分位范围定位；扩大边长到 1.25 倍，并保留至少原图短边的 45%。绿色不足时退回整图。该规则不读取类别标签，训练与测试使用同一规则。

这是一项定位裁剪对照，没有使用人工分割标注。原始图像不被覆盖。保存全部裁剪坐标、源图与裁剪图哈希及接触表；在裁剪图上重新提取冻结 DINOv2 多层 CLS 和 patch 表征，并进行嵌套五折分类。

代码：`outputs/plant_prepare_crops.py`、`outputs/plant_v4_crop_features.py` 和已有的 `outputs/plant_v3_features.py`。实际提取时逐像素确认未改变的 653 张图，复用其冻结特征，只重新计算改变的 225 张图。

## 3. 生物学预训练表征

[BioCLIP 2（NeurIPS 2025）](https://arxiv.org/html/2505.23883v2)采用带分类层级的生物图像预训练，论文分析了物种区分及种内生命阶段差异的表征。这里的动机是检查这种预训练能否改善相近幼苗识别；论文的其他基准成绩不能直接预测本数据的成绩。

本轮候选是后续发布的 [BioCLIP 2.5 Huge 官方模型](https://huggingface.co/imageomics/bioclip-2.5-vith14)，ViT-H/14，默认 224 输入；它是 2026 年模型版本，不能把它写成 2025 年论文的原始 ViT-L 模型。使用 [作者仓库](https://github.com/Imageomics/bioclip-2)对应的 OpenCLIP 实现。权重固定 revision `6e3d04e3d6522012c88181085c5ae666e14c45cd`，官方 safetensors SHA-256 为 `ac2e37c2f89ef8e6b889176a9a3f418970ad9db15a218bd29e3321e95c46ae97`；通过完整校验后才加载。

实现只加载图像编码器。比较投影后图像向量、多层 CLS+全局 patch、多层 CLS+绿色 patch；分类头使用平均视角或逐视角训练，内层四折选择 C∈{0.1,1,10}。图像骨干保持冻结。这是将官方模型适配本实验，未复现大规模预训练，也未引入额外有标签幼苗图片。

代码：`outputs/plant_bioclip_probe.py`。权重下载成功、完整五折完成之前，不声称已获得该模型的分类结果。

## 4. BioCLIP 的小样本微调

[LP-FT（ICLR 2022 Oral）](https://arxiv.org/abs/2202.10054)提出先拟合线性分类头、再微调，以缓解迁移学习中的预训练特征扭曲。本实验借鉴这种初始化方式：每个外层折先仅用该折训练样本拟合 StandardScaler 和逻辑回归，将二者严格换算为线性层，再训练 BioCLIP 最后两个 Transformer block、末端 LayerNorm 和分类头。其余层冻结。

预先固定 15 个 epoch，输入 224，batch=4、梯度累积=2，骨干学习率 1e-5、分类头 1e-4、AdamW weight_decay=0.05。增强包含轻裁剪、90 度旋转、水平翻转、轻色彩扰动及 25% 概率的 MixUp；标签平滑 0.03。使用固定末轮 EMA（衰减 0.98），不按外层验证集挑 epoch。最终对五折模型、八种视角的概率平均。这是 LP-FT 思路的任务适配，不是论文中全量微调的原样复现。

为适配 8GB 显存，冻结部分保留 FP16，所有可训练参数保持 FP32，前向使用 AMP。实现为 `outputs/plant_bioclip_lpft.py`，复用 `outputs/plant_finetune.py` 的训练循环。

## 5. 非线性分类头对照

在已经提取的 DINOv2/BioCLIP 前景特征上构造 RBF 核，增加 SVM 对照。每张图单独进行特征归一化；固定 γ∈{1,10,50}、C∈{1,10,100}、BioCLIP 核混合比例∈{0,0.5,1}，通过内层交叉验证选择。所有 SVM 拟合及其概率校准只读取当前训练子集。核矩阵的非训练行列不进入拟合。代码为 `outputs/plant_v4_kernel.py`。

## 6. 邻域一致性对照

[LaplacianShot（ICML 2020）](https://arxiv.org/abs/2006.15486)及[作者代码](https://github.com/imtiazziko/LaplacianShot)利用查询图像之间的邻域一致性进行传导推断。另检索了 [TransCLIP（NeurIPS 2024）](https://proceedings.neurips.cc/paper_files/paper/2024/file/71d7dbe2652bd4662d29fa269f059db4-Paper-Conference.pdf)，但本轮没有实现其高斯模型和 KL 约束优化，不能声称复现 TransCLIP。

本轮实际增加的是简单的固定图传播对照：用冻结 DINOv2 前景描述构建包含训练节点和查询节点的余弦 5 近邻图，温度 0.1，行归一化。查询节点初值为原 LP-FT 概率，训练节点为 one-hot 真值；迭代 `P_new=(1-alpha)*P_initial+alpha*W*P`，每轮重新固定训练节点，最多 100 轮、变化小于 1e-8 时停止。仅比较事先固定 alpha=0.2、0.5 两个值。每个外层验证折单独建图且不读验证标签；最终图只读取 500 个训练标签和 378 张测试图的无标签特征。这是邻域一致性思路的适配，不是上述论文原算法复现。推断依赖查询批次组成；不强制查询类别均衡。

## 7. 方案选择

在 BioCLIP 完整结果前固定 11 个候选，随后在核分类器结果前加入其单模型和与原模型等权平均两个候选，再在图传播结果前加入两个固定参数图传播候选，共 15 个。基础部分包括六个单独模型（原模型、高分辨率、裁剪、冻结 BioCLIP、微调 BioCLIP、核分类器），原模型分别与五个新模型的等权平均，以及原模型/裁剪/冻结 BioCLIP、原模型/裁剪/微调 BioCLIP 两个三等分平均。按 OOF 准确率优先、log-loss 次之选择，不扫描连续融合权重。该选择仍复用了开发折，其最优成绩不能看作独立验证分数。

## 复现入口

从项目根目录运行：

```powershell
python outputs/plant_v4_resolution.py --size 518 --batch-size 4 --output-dir outputs/stronger_v4_20260929/dino_518
python outputs/plant_prepare_crops.py
python outputs/plant_v4_crop_features.py
python outputs/plant_v3_features.py evaluate --run-dir outputs/stronger_v4_20260929/dino_crop
python outputs/download_bioclip.py
python outputs/plant_bioclip_probe.py extract
python outputs/plant_bioclip_probe.py evaluate
python outputs/plant_bioclip_lpft.py
python outputs/plant_v4_kernel.py
python outputs/plant_v4_graph.py
python outputs/select_plant_v4.py
python outputs/finalize_plant_v4.py
```

完整结果、最终方案及新提交哈希将在本轮实际运行完成后写入 `RESULTS.md`。0.9735 仍对应 v3 文件。
