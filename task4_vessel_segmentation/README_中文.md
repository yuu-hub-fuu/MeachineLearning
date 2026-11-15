# 任务4：眼底血管分割

本工程只使用 `2026task-4.zip` 中的20张训练图及其标签。从零训练自行实现的 U-Net，不使用外部训练数据、外部眼底标签或预训练权重。测试集只用于最后推理。

## 文件与复现

所有文件均保存在本项目目录，训练不会新建 C 盘模型缓存。

```powershell
cd 'D:\新建文件夹\websites\task4'
python -u -X utf8 pipeline.py smoke --smoke-steps 150 --eval-every 50
python -u -X utf8 pipeline.py train --steps 1800 --eval-every 150
python -u -X utf8 predict_deterministic.py
```

`train` 会依次完成五折训练，并自动进行五模型集成推理、导出和提交格式校验。训练会覆盖对应折的模型，复现前请先备份当前 `runs`。只用已交付权重重新推理：

```powershell
python -u -X utf8 predict_deterministic.py
python -X utf8 pipeline.py validate
```

可单独指定待训练的折，例如 `--folds 2 3 4`。程序不自动恢复折内优化器状态，而是从该折头部重新训练；其他折权重保留。完整训练后自动推理必须满足五折权重齐全。

Python 3.13、NVIDIA CUDA GPU。已验证环境版本见 `requirements.txt`；PyTorch GPU 版需与本机驱动匹配，当前使用 PyTorch 2.6.0+cu124。提供的代码使用 CUDA 混合精度，训练和推理均需要 CUDA。

## 数据、规则与编码

数据共60个图像文件：20张训练图、20张训练标签、20张测试图，均为565×584（宽×高）。解压时检查路径没有逃出目标目录，原压缩包CRC已通过；审计结果见 `audit.json` 和 `data_manifest.json`。

实际训练标签只有0和255，白色是血管。题面写“血管0、其他255”，与实际标签相反；用户已确认优先遵循实际标签和官方转换脚本。主提交 `delivery/submission.csv` 编码血管。`delivery/vessel_white` 是生成主提交的图片；`delivery/vessel_black` 是供查看或按题面交付的反色图片，不能直接拿去生成同一含义的CSV。

官方 `segmentation_to_csv.py` 会先转灰度，用 Pillow 的双线性插值缩放至512×512，再把任意大于0的像素视作前景，最后按行展开，使用从1开始的起点和长度。这种转换会扩大细血管边缘。训练主目标和主验证指标完整复现此转换；另记录最近邻缩放原始标签下的 Dice，两种结果不能混称。

最终预测直接保存为512×512的无损二值PNG，再运行原封不动的官方脚本。只对CSV行顺序按 `sample_submission.csv` 重排，不改变脚本产生的RLE。另用独立的向量化编码器和解码器逐像素核对全部20张，检查正数、起点顺序、范围、重叠、前景比例、ID及反色关系，并覆盖全黑、全白、对角线边界情况。

规则页已于北京时间2026-11-03读取：允许网上已有模型或自己设计模型，不得抄袭同学代码，比赛结束需提交能够复现结果的代码，每日提交次数有限。页面显示截止北京时间2026-11-07 23:59:59。规则来源：<https://www.kaggle.com/competitions/2026task-4/rules>。当前流程只生成本地提交文件，不会上传到Kaggle。

## 验证设计和方法

种子20261103，将20张训练图固定分成五折，每折16张训练、4张验证。按整张图像划分后再裁块，不允许同图裁块跨训练和验证。没有像素完全重复的图像；所有划分保存在 `runs/folds.json` 及各折 `split.json`。

输入为512×512 RGB，加上基于绿色通道局部高斯去背景的第四通道。四通道全部只由当前图像计算，不涉及任何标签统计。网络为四个尺度的U-Net，通道24/48/96/192，双3×3卷积、BatchNorm、ReLU，双线性上采样与跳跃连接，单通道输出。

每折1,800次参数更新，batch=16，随机160×160裁块。增强为同步90度旋转/镜像，亮度和颜色扰动，少量高斯噪声。损失为二元交叉熵加soft Dice，AdamW，初始学习率0.001，余弦衰减至0.00005，weight decay=0.0001，梯度裁剪5，FP16混合精度。

每150步在整张验证图上计算 Dice，按阈值0.5的四图平均 Dice 保存该折最佳权重。最佳权重对每张验证图做0/90/180/270度旋转预测平均，组成每图只由未训练过该图的模型给出的OOF预测。五个模型对测试图分别做相同旋转平均后再取均值。

最终阈值仅使用20张OOF预测，在0.15至0.85、间隔0.025的范围内选择平均 Dice 最大值。`runs/metrics.json` 同时保存固定阈值0.5、阈值选择后分数、全部候选阈值和逐图分数。验证集用于选权重和阈值，因此这些是模型选择阶段的验证估计，并非完全未参与选择的独立测试集分数，更不是Kaggle线上分数。

U-Net结构参考：Olaf Ronneberger, Philipp Fischer, Thomas Brox. *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI 2015. <https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/>。本工程网络代码为本次独立实现，使用的是该结构思路。

## 产物

- `delivery/submission.csv`：最终提交，20行，列名 `Id,Predicted`。
- `delivery/submission_validation.json`：独立回读校验、逐图像素比例和CSV SHA-256。
- `delivery/vessel_white`：实际标签语义的白血管二值图。
- `delivery/vessel_black`：题面文字语义的黑血管二值图。
- `delivery/test_preview.png`：测试原图、预测、叠加图。
- `delivery/validation_preview.png`：OOF最差五张，依次为原图、标签、预测、错误图；错误图绿色TP、红色FP、蓝色FN。
- `runs/fold0` 至 `runs/fold4`：最佳权重、验证预测、逐轮日志、划分和汇总。
- `runs/config.json`：正式训练参数、软件版本、代码和官方脚本哈希。
- `runs/probabilities.npz`：OOF及测试概率图，方便回查与审计。
- `runs/train.log`、`runs/status.json`：训练全过程日志和最新状态。

训练采用固定种子，但CUDA卷积算法的选择和不同软硬件可能带来微小差异。最终导出使用 `predict_deterministic.py` 固定推理算法，并在解压后的工程中重跑核对CSV。`pipeline.py infer` 保留为原始训练程序中的快速推理入口；训练结束的自动导出之后，运行确定性入口生成最终交付CSV。不承诺不同GPU从零训练得到逐位一致结果。
