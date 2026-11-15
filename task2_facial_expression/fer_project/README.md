# 2026task2 六类人脸表情分类

**当前交付主线为 Dima ViT 主干微调与推理时适配。** 最终配置见上级目录 `model_selection.json`、`实验结果.md`，预测复现入口是 `infer_dima_finetuned.py`。下文 B0、多模型搜索等内容保留为实验历史，不是当前推荐重跑的完整队列。全部模型和缓存位于 `D:\FER2026task2`。

本项目使用同一份一次留出验证划分和唯一随机种子 `42`。不做 K 折交叉验证，不做多种子训练。

标签严格固定为 `Angry=0, Fear=1, Happy=2, Sad=3, Surprise=4, Neutral=5`。最终 CSV 列为 `ID,Emotion`，顺序严格跟随原始 submission.csv。

## 数据与验证

原始数据：28,275 张有标签训练图、7,065 张无标签测试图，全部是 48×48 灰度图。为比较方案，仅从 train 留出一次约 15% 的验证集：24,019 张用于训练、4,256 张用于验证。同像素重复图按组划分，不跨训练/验证边界；39 组训练标签冲突保留原标签并保持整组在同一侧。

验证准确率用于选 checkpoint，Macro-F1、每类召回与混淆矩阵用于分析。test 不用于调参；其输出为深度模型的预测。选定方案后可用全部 train 按验证选择的轮数重新训练，得到最终提交。

## 方法设计与来源

骨干采用 EmotiEffLib 的表情领域预训练 EfficientNet-B0。作者的训练代码采用 `tf_efficientnet_b0_ns`；在当前 timm 中使用对应名称 `tf_efficientnet_b0.ns_jft_in1k`，并严格载入全部原始参数。原始八类分类器取 `[0,3,4,6,7,5]` 行初始化本题的六类分类器。

- B0 `baseline`：表情预训练骨干直接微调。
- B1 `local`：借鉴 EfficientFace（AAAI 2021）的局部—全局协同，在骨干 stride=8 的 40×20×20 中间特征图上划分四个区域，分别使用深度卷积提取局部信息，再与最终 1280 维全局特征融合。使用中间层保留局部空间信息，避免仅在末端 5×5 特征图上切块。当前实现是适配 EfficientNet 的改造，并非 EfficientFace 完整复现。
- B2 `scn`：B1 + SCN（CVPR 2020）的 sigmoid 样本重要性分支，以重要性缩放 logits，并加入高/低重要性组的排序约束。比例 0.7，margin 0.07；保留原标签，不启用原论文的硬重标注。
- B3 `rul`：B1 + RUL（NeurIPS 2021）式相对不确定性特征混合。学习正的不确定性，按两个样本的不确定性比例混合特征，使用两标签等权 add-up loss。当前实现保留原图 CE 并以 0.5 系数加入混合损失，属于改造版本。

只改变被比较的模块。四个方案统一使用输入 160×160（原图信息仍来自 48×48；插值不创造细节）、batch 64、AdamW、骨干初始学习率 1e-4、新增/分类层学习率 5e-4、weight decay 1e-4、两轮 warmup + cosine、label smoothing 0.05、水平翻转、±10° 小角度仿射和轻度擦除。初始计划每方案 12 轮；以实测吞吐量和验证曲线决定是否统一扩展训练预算。单独保存无 TTA 与水平翻转 TTA 的验证指标。

DMUE（CVPR 2021）的潜在标签分布学习和 FaceXFormer（ICCV 2025）的任务 token/双向交叉注意力作为设计比较候选，未在上述四个实验中实现。POSTER 是 ICCV Workshop 论文，不列为 CCF A 主会依据。所有改造的有效性均待本数据实验验证，论文基准分数不等于本比赛成绩。

论文与作者代码：

- [EfficientFace / AAAI 2021](https://aaai.org/papers/03510-robust-lightweight-facial-expression-recognition-network-with-label-distribution-training/)，[作者代码](https://github.com/zengqunzhao/EfficientFace)。已阅读 `models/EfficientFace.py` 中四区域局部提取与全局分支融合实现。
- [SCN / CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Suppressing_Uncertainties_for_Large-Scale_Facial_Expression_Recognition_CVPR_2020_paper.html)，[作者代码](https://github.com/kaiwang960112/Self-Cure-Network)。已阅读 `src/train.py` 的重要性缩放、排序损失和重标注逻辑。
- [RUL / NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/9332c513ef44b682e9347822c2e457ac-Paper.pdf)，[作者代码](https://github.com/zyh-uaiaaaa/Relative-Uncertainty-Learning)。已阅读 `src/rul.py`、`src/main.py`、`src/utils.py`。
- [DMUE / CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/She_Dive_Into_Ambiguity_Latent_Distribution_Mining_and_Pairwise_Uncertainty_Estimation_CVPR_2021_paper.html)，[作者代码](https://github.com/JDAI-CV/FaceX-Zoo/tree/main/addition_module/DMUE)。
- [FaceXFormer / ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Narayan_FaceXFormer_A_Unified_Transformer_for_Facial_Analysis_ICCV_2025_paper.html)，[作者代码](https://github.com/Kartik-3004/facexformer)。
- [EmotiEffLib 作者代码与预训练权重](https://github.com/sb-ai-lab/EmotiEffLib)。使用其 `enet_b0_8_best_vgaf.pt`；其分类顺序及标准化见 `emotiefflib/facial_analysis.py`。

## 路径与运行

Python 环境为本机 Python 3.13、PyTorch 2.6.0+cu124、torchvision 0.21.0+cu124、timm 1.0.29。GPU 为 RTX 4060 Laptop 8GB。

所有预训练权重与模型缓存均在 D 盘：

```text
D:\FER2026task2\pretrained\enet_b0_8_best_vgaf.pt
D:\FER2026task2\cache\
D:\FER2026task2\data\dataset.npz
D:\FER2026task2\runs\
```

在本目录运行（首次运行需要先准备指定的官方权重）：

```powershell
py -3.13 -X utf8 fer.py prepare
py -3.13 -X utf8 prepare_weights.py
py -3.13 -X utf8 verify.py
py -3.13 -u -X utf8 fer.py train --variant baseline --name baseline_s42_w0 --epochs 12
py -3.13 -u -X utf8 fer.py train --variant local --name local_s42_w0 --epochs 12
py -3.13 -u -X utf8 fer.py train --variant scn --name scn_s42_w0 --epochs 12
py -3.13 -u -X utf8 fer.py train --variant rul --name rul_s42_w0 --epochs 12
```

也可运行 `py -3.13 -u -X utf8 run_experiments.py --epochs 12`，按顺序完成四方案及最终全量训练。选择规则预先固定为验证 TTA 准确率最高的方案；仅在最好的两个方案等权融合提升至少 0.25 个百分点时，重训两个方案并等权融合。否则只重训最佳单模型。全部重训仍只使用 seed=42，保持与对应验证实验相同的轮数和训练设置。使用同一留出集选模型后的分数存在选择偏差，不声称它是无偏泛化估计。

每个运行目录包含配置、划分摘要、逐轮日志、最佳与最后 checkpoint、验证集预测与指标、测试概率及 submission.csv。相同运行名称不会覆盖已有日志。`--full` 用于最终全 train 重训；轮数应取此前验证选定值，不在全量训练后伪造验证指标。

独立推理示例（按最终 `model_selection.json` 的 `final_runs` 选择 checkpoint；融合时给出两个路径）：

```powershell
py -3.13 -X utf8 fer.py infer --checkpoint D:\FER2026task2\runs\baseline_s42_w0\best.pt --output D:\FER2026task2\submission_reproduced.csv
```

独立推理直接严格加载训练后的 state_dict，无需重新下载原始预训练权重。

本机有 16GB 系统内存，正式队列采用 `num_workers=0`，避免 Windows 多进程重复加载运行库引发内存压力。最初的 `baseline_s42` 启动在完成首轮前中断，仅保留为启动诊断；正式比较使用 `*_s42_w0`。

比赛每天最多提交五次；本代码生成 CSV，不自动提交、不修改 Kaggle 组名。

## 错误样本、特征与推理优化

`optimize_predictions.py` 在四方案与全量重训结束后运行，固定已选网络，只比较有限的推理设置：

1. 单视图；原图加水平翻转；144/160/176 三尺度各加翻转；前三尺度加裁去四边各一个像素的轻微中心裁剪及其翻转。分别为 1、2、6、8 个视图，均平均 softmax 概率。增加视图不保证提升。
2. 提取 1280 维归一化特征。局部分支模型另外保存全局、局部特征供分析；近邻使用融合特征。特征与视图缓存全部存放在 `D:\FER2026task2\analysis_cache`，由 checkpoint SHA256 和数据划分指纹区分。
3. 固定余弦 kNN 的 k=15、温度=0.07，只尝试网络概率与近邻概率按 0.1/0.2 权重融合。验证时特征库仅含 24,019 张训练子集图；选定后，最终全量模型可使用全部 28,275 张有标签训练图作为特征库。

TTA 相对原始翻转方案、特征融合相对已选 TTA，均要求验证准确率提高至少 **0.25 个百分点**才采用。该门槛是预先固定的实用阈值，不代表统计显著。所有方法仍共用一次验证划分；存在多次选择偏差，不能据此声称 Kaggle 分数或无偏泛化性能。

错误分析输出位于 `error_analysis/`：高置信验证误分类表、前三近邻图册、训练复查候选表、完全相同像素但标签冲突的分组。训练复查候选需同时满足：模型预测与原标签不同、预测置信度至少 0.9、原类中心的余弦距离弱于其他类中心。训练预测是样本内诊断；这些条件不证明标签错误。**不删除或覆盖任何原始标签，也不修改验证标签。** SCN/RUL 实验用于比较对不确定样本的鲁棒训练是否有益。

从零完成全部流程：

```powershell
py -3.13 -u -X utf8 finish_pipeline.py
```

已有四方案和全量模型时，只重新运行优化与打包：

```powershell
py -3.13 -u -X utf8 optimize_predictions.py
py -3.13 -X utf8 summarize.py
```

`fer.py infer` 重现基础翻转 TTA；若最终选择了额外 TTA/kNN，请运行 `optimize_predictions.py` 重现最终提交。它读取 `model_selection.json` 和对应 checkpoint，可复用有指纹的缓存。`submission_core.csv` 保留未加后处理的原始输出；`submission.csv` 始终是最后通过验证选择的版本。`optimization_results.csv` 记录全部候选、最终设置及增益；`model_selection.json` 记录最终来源。

流水线状态在 `D:\FER2026task2\runs\finish_pipeline_status.json`，阶段日志分别在 `run_experiments` 的各模型日志、`optimize_predictions.log`、`upgrade_experiments.log`、`summarize.log`。失败时状态写为 `failed`，不会把失败当作完成。最终打包前检查全部 7,065 个 ID 的顺序、合法标签、概率归一化和 CSV 与概率 argmax 一致性。

## 强预训练与特征适配升级

论文筛选依据、作者仓库、方法边界见上级目录 `论文检索与升级方案.md`。
新增 NLA（AAAI 2025）作者 RAF-DB 权重、QCS（AAAI 2025）作者 FERPlus 权重，以及 RMN（ICPR 2020）作者 FER2013 权重。**后两者可能与本题存在源图像重叠，分数不代表独立泛化。** 没有使用外部源图像标签回填比赛测试答案。

准备权重和 RMN 的独立图像依赖（所有下载缓存和安装位置均为 D 盘）：

```powershell
$env:TEMP='D:\FER2026task2\cache\tmp'
$env:TMP=$env:TEMP
$env:PIP_CACHE_DIR='D:\FER2026task2\cache\pip'
py -3.13 -m pip install --no-deps --target D:\FER2026task2\dependencies opencv-python-headless==4.11.0.86
py -3.13 -u -X utf8 prepare_upgrade_weights.py
py -3.13 -u -X utf8 upgrade_experiments.py --smoke nla
py -3.13 -u -X utf8 verify_upgrade.py
```

`finish_pipeline.py` 现按基础四方案→基础 TTA/特征分析→强预训练特征适配→校验打包运行。应先完成权重下载。已有基础结果时可单独运行：

```powershell
py -3.13 -u -X utf8 upgrade_experiments.py --kinds nla qcs rmn --epochs 20 --batch 16
py -3.13 -X utf8 summarize.py
```

升级阶段冻结图像骨干，原图/翻转特征缓存在 `D:\FER2026task2\upgrade_cache`。比较原始六类头、线性适配、CE 残差适配、SupCon+标准概率 JSD 残差适配。每个版本仅 seed=42，最多 20 个特征训练 epoch，使用同一分组验证。残差适配器为 LayerNorm→128 维→原特征维度，末层零初始化、残差系数 0.1，原分类头初始化保留。CE label smoothing=0.02，SupCon 权重 0.05（前两轮不启用），概率 JSD 权重 0.1。

可靠度仅调节对比损失的 anchor 权重，取 `0.25+0.75*p_teacher(原始标签)`；所有样本保留完整 CE。它是本任务的简化设计，**并非 NLA 原文的二维高斯建模复现**。QCS 使用作者已训练的单图推理分支，未复现四分支训练。

最终按验证选定轮数全量重训适配器。升级优于原方案至少 0.25 个百分点才替换 `submission.csv`，此前版本保存在 `before_upgrade_*`。若升级被采用，`model_selection.json` 的 `upgrade.selected` 和 `upgrade.final_checkpoint_paths` 才是最终来源；顶层 `final_runs` 与 `postprocessing` 保留基础阶段记录。

升级的独立推理（无需重新训练）：

```powershell
py -3.13 -u -X utf8 infer_upgrade.py --output D:\FER2026task2\submission_upgrade_reproduced.csv
```

`fer.py infer` 和 `optimize_predictions.py` 只适用于 B0 阶段；**升级后不要用它们覆盖最终提交**。升级的错误表为 `upgrade_validation_errors.csv`，旧 `error_analysis/` 属于 B0。作者源文件和 MIT 许可证保存在 `vendor/`；本地兼容改动见 `vendor/ADAPTATIONS.md`。

## Hugging Face 专用权重与优先队列

补充三组已微调的 ViT：`dima806/facial_emotions_image_detection`、`mo-thecreator/vit-Facial-Expression-Recognition`、`trpakov/vit-face-expression`。运行 `prepare_hf_weights.py` 下载固定 revision 的 safetensors、预处理和类别配置，保存到 D 盘并核对发布的 SHA256。对应本地代号 `hf_dima`、`hf_mo`、`hf_trpakov`。筛选依据见上级目录 `HF模型筛选.md`。

本次实际执行优先队列：完成 baseline/local/scn 后暂缓 RUL 和 B0 全量重训，先保存有效的临时提交，再比较强预训练模型。仅当升级均未超过基础模型时才回到基础模型全量重训。这是算力优先级调整，不把未运行的 RUL 计入结果。

完整升级比较命令（下载须先完成；运行中按已下载可用的模型先行处理）：

```powershell
py -3.13 -u -X utf8 prepare_hf_weights.py
py -3.13 -u -X utf8 upgrade_experiments.py --kinds hf_dima rmn qcs hf_mo hf_trpakov nla --wait-weights-seconds 3600
py -3.13 -u -X utf8 summarize.py
```

升级完成前已有的 `submission.csv` 可能仍是已完成基础模型的临时版本，是否正式采用升级以 `model_selection.json` 的 `upgrade.adopted_over_core`、`submission_verification.json` 和流水线 `completed` 状态为准。

## 当前主线：Dima 主干微调（2026-10-18）

用户已停止扩大模型搜索，接受外部预训练潜在数据重叠，要求基于 Dima 进行论文方法训练。旧的多模型 priority_pipeline 已停止，不要重新运行它。

```powershell
py -3.13 -u -X utf8 finetune_dima.py --all
py -3.13 -u -X utf8 infer_dima_finetuned.py
py -3.13 -u -X utf8 summarize.py
```

`finetune_dima.py` 顺序运行 CE、SupCon+一致性两组主干微调，分别从原始 Dima 初始化，解冻最后 4 个 Transformer 块。仅 seed=42，同一分组验证；每组最多 5 轮、有早停和每轮断点恢复。获选设置从原始权重在全量训练集重训，再生成 CSV；若没有超过已有线性适配，则保留并全量重训线性方案。具体设计与论文对应关系见 `../Dima主干微调方案.md`。

此阶段最终选择以 `model_selection.json` 的 `dima_finetune` 和 `dima_finetune_results.json` 为准。`infer_dima_finetuned.py` 可从 D 盘公开权重及可训练参数差量独立复现。不要用旧 B0 推理脚本覆盖最终文件。所有分数仍是本地验证，存在外部预训练重叠风险，不代表 Kaggle 分数或无污染泛化结果。

## 推理时无标签微调

训练完成后运行 `tta_dima.py` 比较 MEMO-LN 和可靠样本熵最小化（EATA 思想），只更新 13,824 个 LayerNorm 参数，每批两步后恢复，固定 seed=42 打乱图像后组批。未使用验证/测试标签计算梯度；验证标签只用于比较是否采用。

运行顺序：`finetune_dima.py --all` → `tta_dima.py` → `analyze_dima.py` → `analyze_dima_features.py` → `infer_dima_finetuned.py` → `summarize.py`。最终是否启用推理适配由 `dima_tta_results.json` 的 adopted 决定，推理入口会自动使用相同配置。完整描述见 `../推理时适配方案.md`。
