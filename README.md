# 机器学习课程设计：四个视觉任务

| 目录 | 任务 | 方法 | 私榜成绩 |
|---|---|---|---|
| `task1_plant_classification` | 植物幼苗五分类 | DINOv2-L 最后四层 LP-FT（等价线性头初始化），绿色先验前景加权表征，8 视角 TTA，5 折集成 | 准确率 0.9735 |
| `task2_facial_expression` | 人脸表情六分类 | Dima ViT-B/16 最后四层局部解冻微调，层间学习率衰减 | 准确率 0.828 |
| `task3_fovea_localization` | 眼底黄斑中央凹定位 | HRNet-W32 + DARK 亚像素热图定位，眼底视野自适应裁剪，5 折模型平均 | RMSE 20.96089 |
| `task4_vessel_segmentation` | 眼底血管分割 | 评价算子感知的 TransferUNet（ImageNet ResNet-18 + 全分辨率细节分支），与公开 RLAD 模型融合 | Dice 0.86403 |

各任务目录下的 README 说明了数据放置方式、训练与推理命令。

## 主要入口
- 任务一：`plant_classifier.py`（数据核验、冻结特征与嵌套交叉验证）、`plant_dino_lpft.py` / `plant_finetune.py`（LP-FT 微调）、`predict_plant_dino.py`（推理）。
- 任务二：`fer_project/fer.py`（数据准备与基础模型）、`fer_project/finetune_dima.py`（主干微调）、`fer_project/infer_dima_finetuned.py`（推理）。
- 任务三：`main.py`（`prepare` / `train` / `submit-file`）。
- 任务四：`pipeline.py`（V1）、`v4_transfer.py`（V4 训练）、`predict_v4.py`、`predict_open_rlad.py`（最终融合推理）。

## 不包含的内容
数据集、模型权重、特征缓存与训练日志中的大文件未上传；第三方代码保留其原始许可证。
