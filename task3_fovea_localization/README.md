# 任务3：眼底黄斑中央凹定位（HRNet-W32 + DARK）

## 方法
- 预处理：按亮度阈值求眼底视野的外接正方形，裁剪后缩放到 384×384（不使用任何标注或检测框）。
- 网络：HRNet-W32（ImageNet 预训练），单关键点，输出 96×96 热图。
- 监督：以真实坐标为中心、sigma=2 的浮点高斯热图（DARK 编码），JointsMSE 损失。
- 解码：DARK 亚像素解码，逆变换回原图坐标；推理时与水平翻转图的热图取平均。
- 训练：AdamW，骨干学习率 2e-4、输出层 1e-3，余弦衰减，40 轮，混合精度，取最后一轮权重。
- 增强：旋转 ±20°、缩放 ±15%、水平翻转、颜色扰动。
- 验证：按分辨率与中央凹左右侧分层、并把重复/同源图分到同组的 5 折（StratifiedGroupKFold，种子 20261027）。

## 结果
| 提交 | 公开榜 RMSE | 私榜 RMSE |
|---|---|---|
| 5 折模型预测坐标平均（`submit-file` 命令生成） | 16.98771 | 20.45190 |

本地 5 折 OOF RMSE 为 18.31。

## 运行
```bash
pip install -r requirements.txt
python main.py --root work prepare --archive 2026task3.zip
python main.py --root work train --name hrnet_s384
python main.py --root work submit-file --checkpoints work/runs/hrnet_s384/fold0/best.pt work/runs/hrnet_s384/fold1/best.pt work/runs/hrnet_s384/fold2/best.pt work/runs/hrnet_s384/fold3/best.pt work/runs/hrnet_s384/fold4/best.pt --output submission.csv
```
`--root` 下需放 HRNet-W32 ImageNet 权重 `cache/hrnet_w32-36af842e.pth`（来源：TransPose 作者发布页 https://github.com/yangsenius/TransPose/releases/tag/ImageNet）。

## 第三方代码
`third_party/DarkPose` 为 DARK 作者开源实现（https://github.com/ilovepose/DarkPose ，commit 0185b427f34a08d93ea7e948ccc3e7fdc78d88c5，MIT 许可，见其中 LICENSE），未作修改。复用其中的 HRNet 网络、JointsDataset 数据增强与高斯目标、JointsMSELoss 和 DARK 解码；本仓库代码负责数据准备、分组交叉验证、训练组织与提交文件生成。

参考文献：
- Sun et al. Deep High-Resolution Representation Learning for Human Pose Estimation. CVPR 2019.
- Zhang et al. Distribution-Aware Coordinate Representation for Human Pose Estimation. CVPR 2020.
