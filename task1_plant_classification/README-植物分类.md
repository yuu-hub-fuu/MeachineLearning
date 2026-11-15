# 五类植物物种识别

本实现针对提供的数据目录：5 个物种、每类 100 张训练图像，以及一份 378 行测试提交清单。任务标签是物种类别；生长阶段作为类内外观变化，不单独预测。

## 方法

采用 Meta DINOv2 冻结预训练骨干，按官方线性评估实现拼接最后 4 个 block 的 CLS token，并加入末层 patch token 均值；原图和水平翻转图的特征取平均。分类头在类别平衡的逻辑回归、线性 SVM、RBF-SVM 中选择，所有缩放均在交叉验证折内拟合。嵌套分层交叉验证的外层折只评估、内层折选分类头与超参数，最终再以全部训练图拟合提交模型。DINOv2 权重保持冻结，以避免 500 张小样本上全量微调造成过拟合。

DINOv2 论文及官方仓库说明其自监督预训练视觉特征可直接用于线性分类器；官方评估实现提供多层 CLS 表征拼接选项。[论文](https://arxiv.org/abs/2304.07193) · [官方仓库](https://github.com/facebookresearch/dinov2)

植物任务的直接证据包括：Ong 等人在 PlantCLEF2020 上研究 DINOv2 植物跨域识别，发现冻结特征加简单分类头具有竞争力，同时指出少样本类上的泛化仍是限制；Gustineli 等人的 PlantCLEF 2024 工作同时利用基础和微调 DINOv2 特征，但其训练数据规模约 140 万张，与本实验 500 张相差很大。因此本流程先以交叉验证检验不同骨干，不把大规模数据上的微调结果直接当成本任务结论。[Ong et al., 2025](https://doi.org/10.1007/s00521-025-11499-6) · [Gustineli et al., CLEF 2024](https://arxiv.org/abs/2407.06298)

## 数据目录

将 `2026task1.zip` 解压后，包内实际目录名为 `dataset-for-task2/dataset-for-task2/`：

```text
dataset-for-task2/
  dataset-for-task2/
    train/
      Black-grass/             # 100 张
      Common wheat/            # 100 张
      Loose Silky-bent/         # 100 张
      Scentless Mayweed/        # 100 张
      Sugar beet/               # 100 张
    test/                       # 378 张
  submission-for-task2.csv     # ID,Category
```

提交清单中的 378 个 ID 与测试图片完全对应；其中现有 `Category` 值都是 `Sugar beet` 占位标签，预测时会被模型输出替换。

## 数据核验

先将 ZIP 解压到一个目录，然后核对类别数、每类图片数和测试 ID：

```powershell
python plant_classifier.py audit `
  --train-dir "D:\path\to\dataset-for-task2\dataset-for-task2\train" `
  --test-dir "D:\path\to\dataset-for-task2\dataset-for-task2\test" `
  --input-csv "D:\path\to\submission-for-task2.csv"
```

## 依赖

Python 3.10+、PyTorch、torchvision、Pillow、NumPy、scikit-learn、joblib。首次调用会通过 PyTorch Hub 下载 DINOv2 代码和权重，并缓存到本机。建议用支持 CUDA 的 PyTorch。默认批量大小为 2；脚本对原图和翻转图同时提特征，因此实际送入骨干的图片批量是参数值的两倍。RTX 4060 8GB 建议先保持 batch 2，若仍显存不足可使用 `dinov2_vits14` 或降低 `--image-size`。

## 训练分类头与交叉验证

```powershell
python plant_classifier.py train `
  --train-dir "D:\path\to\dataset-for-task2\dataset-for-task2\train" `
  --checkpoint "plant_dinov2_vitl14.joblib" `
  --model dinov2_vitl14 `
  --batch-size 1 `
  --amp
```

此命令提取冻结骨干特征、执行嵌套分层交叉验证并保存最终分类器；它不会微调 DINOv2 权重。

## 生成提交文件

```powershell
python plant_classifier.py predict `
  --input-csv "D:\path\to\submission-for-task2.csv" `
  --image-dir "D:\path\to\dataset-for-task2\dataset-for-task2\test" `
  --checkpoint "plant_dinov2.joblib" `
  --output-csv "submission.csv" `
  --batch-size 2 `
  --amp
```

预测阶段核对测试图片是否齐全、ID 是否重复、输出类别是否属于 5 个训练类别，并保持输入 CSV 行序与 `ID,Category` 两列格式。

## 第一版训练结果（2026-09-24）

ViT-B/14 改进版在 500 张训练图上的嵌套 5 折 balanced accuracy 为 **0.9260 +/- 0.0270**，较原单层 CLS+patch 特征基线 **0.9080 +/- 0.0383** 有提升。继续对比官方 ViT-L/14 后，5 个外层折分数为 0.980、0.920、0.890、0.960、0.970，均值为 **0.9440 +/- 0.0378**；全训练集内层交叉验证选择逻辑回归（C=10.0）。ViT-L 分数均值更高，但折间波动也较大。2026-09-29 用户确认这一版 `submission.csv` 在线分数约为 **0.93**，精确小数未知。交叉验证只估计训练集内泛化，不能代替测试集真值或保证线上成绩。

RTX 4060 本地已完成 ViT-B/14 与 ViT-L/14 的特征提取、嵌套交叉验证和全训练集分类头拟合。当前主提交文件 `submission.csv` 使用 ViT-L/14；ViT-B/14 候选保存在 `submission_multilayer.csv`。对应检查点为 `plant_dinov2_vitl14.joblib`、`plant_dinov2_multilayer.joblib`。两份提交都保持 378 行及原始 ID 顺序。

## 第二轮改进（2026-09-29）

新增绿色区域 patch 汇聚、八视角分类头增强及 DINOv2 最后四层 LP-FT 微调。论文依据、复现命令和方法边界见 [研究记录](stronger_20260929/RESEARCH.md)。第二轮文件写入 `stronger_20260929/`，第一版 `submission.csv` 保留，以便对应已知线上 0.93 的结果。

本轮已完成：最终 DINOv2 LP-FT 五折准确率 **0.960**，各折 0.96、0.95、0.96、0.96、0.97；三种固定等权融合未超过它。[submission_v3_20260929.csv](stronger_20260929/submission_v3_20260929.csv) 包含 378 行，较旧版改变 14 张预测。用户随后报告该版线上 **0.9735**，未独立登录平台核验。详见 [完整实验结果](stronger_20260929/RESULTS.md)。

## 第三轮改进（v4，2026-09-29）

增加高分辨率、前景裁剪、BioCLIP 2.5 Huge 冻结特征和局部微调、非线性核分类器。方法和复现命令见 [v4 研究记录](stronger_v4_20260929/RESEARCH.md)，结果将于全部实验结束后写入该目录的 `RESULTS.md`。v3 文件继续保留以对应线上 0.9735。
