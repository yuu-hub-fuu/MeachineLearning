# Hugging Face 表情专用权重筛选

2026-10-18。以下都已针对面部表情任务微调，不是仅有 ImageNet 分类能力的通用初始化。模型卡数字来自发布者，不能直接当作本比赛成绩。

| 本地名称 | 模型 | 发布者报告 | 采用理由 |
|---|---|---|---|
| hf_dima | [dima806/facial_emotions_image_detection](https://huggingface.co/dima806/facial_emotions_image_detection) | 准确率 90.92%，模型卡报告的 25,170 个评估样本近似按七类均衡分布 | 优先验证的表情专用 ViT；必须实测本题六类及当前划分 |
| hf_mo | [mo-thecreator/vit-Facial-Expression-Recognition](https://huggingface.co/mo-thecreator/vit-Facial-Expression-Recognition) | 模型卡摘要 84.34%，训练表最后一项 84.86%；使用 FER2013、MMI、AffectNet | 多来源表情预训练，可能提供互补特征；摘要与最后一轮不是同一个数，不混写 |
| hf_trpakov | [trpakov/vit-face-expression](https://huggingface.co/trpakov/vit-face-expression) | FER2013 验证 71.13%、测试 71.16% | 更接近 48×48 灰度表情域的对照，是否有效由本题验证决定 |

下载保存模型仓库 revision、权重 SHA256、config.json、preprocessor_config.json 和 README。仅加载本地 safetensors，读取 config 中的 id2label 生成六类对应关系，不执行远程自定义代码。不同发布者的类别顺序确实不同，例如 Dima 的顺序是 Sad、Disgust、Angry、Neutral、Fear、Surprise、Happy，不能套用 FER2013 默认顺序。

所有权重和分段临时文件保存在 `D:\FER2026task2\pretrained\hf_*`。新增依赖仅在需要时安装；当前已有 transformers 4.57.3 和 safetensors。

评估流程：原始权重原图/翻转 → 保留原分类头的线性适配 → 残差特征适配 → SupCon 和标准概率 JSD。每个本地训练版本仅 seed=42，仍使用原有一次分组留出验证；公共预训练本身的随机种子以作者信息为准。

先完成可下载权重的验证，再决定是否采用。源预训练可能包含本题的源图像，尤其 FER2013/FERPlus 系列；单独记录来源，不将此验证视为独立泛化评估。测试标签不参与选择，没有获取外部源标签来回填测试图。

后台日志：`D:\FER2026task2\runs\upgrade_experiments.log`。实时起点结果写入 `upgrade_<模型>_screen.json`；最终比较写入 `upgrade_results.json`。模型卡上的高分不会写进本项目实测结果表。
