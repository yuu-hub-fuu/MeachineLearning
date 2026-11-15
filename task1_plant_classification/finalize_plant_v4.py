"""Write an evidence report and reproducibility manifest after all v4 runs."""
import hashlib
import importlib.metadata
import json
import platform
import shutil
from pathlib import Path
import numpy as np
from select_plant_v4 import main as select,ROOT,OLD

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    select()
    report=json.loads((ROOT/'selection_report.json').read_text())
    selected=report['selected'];paired=report['paired_oof']
    names={'baseline336':'v3：DINOv2-L 微调（336）','resolution518':'DINOv2-L 微调（518 推理）',
           'plant_crop':'裁剪 DINOv2-L 冻结特征','bioclip25':'BioCLIP 2.5 冻结特征',
           'bioclip_lpft':'BioCLIP 2.5 最后两层微调','kernel':'DINO/BioCLIP 核分类器',
           'graph_a2':'图传播 alpha=0.2','graph_a5':'图传播 alpha=0.5',
           'baseline_crop_bio':'v3 + 裁剪 + 冻结 BioCLIP',
           'baseline_crop_bio_lpft':'v3 + 裁剪 + 微调 BioCLIP'}
    for key in ('resolution518','plant_crop','bioclip25','bioclip_lpft','kernel'):
        names['baseline_plus_'+key]='v3 + '+names[key]
    rows=['| 候选 | 五折 OOF 准确率 | Log-loss |','|---|---:|---:|']
    rows += [f"| {names[r['name']]} | {r['accuracy']:.3f} | {r['log_loss']:.4f} |" for r in report['candidates']]
    baseline_report=json.loads((OLD/'dino_lpft/cv_report.json').read_text())
    by_class=['| 类别（每类 100 张） | v3 召回率 | v4 召回率 |','|---|---:|---:|']
    for cls in ('Black-grass','Common wheat','Loose Silky-bent','Scentless Mayweed','Sugar beet'):
        by_class.append(f"| {cls} | {baseline_report['classification_report'][cls]['recall']:.2f} | {report['classification_report'][cls]['recall']:.2f} |")
    events=[json.loads(line) for line in (ROOT/'bioclip_lpft/training.jsonl').read_text().splitlines()]
    finals=[e for e in events if 'final_accuracy' in e]
    assert len(finals)==5
    snapshot=ROOT/'source_snapshot';snapshot.mkdir(exist_ok=True)
    sources=['plant_bioclip_probe.py','plant_bioclip_lpft.py','plant_finetune.py',
             'plant_v4_resolution.py','plant_prepare_crops.py','plant_v4_crop_features.py',
             'plant_v3_features.py','plant_classifier.py','plant_dino_lpft.py','plant_v4_kernel.py','plant_v4_graph.py',
             'select_plant_v4.py','finalize_plant_v4.py','download_bioclip.py']
    source_hashes={}
    for name in sources:
        src=Path('outputs')/name;dst=snapshot/name
        if dst.exists() and sha(dst)!=sha(src):raise RuntimeError(f'Snapshot changed: {name}')
        shutil.copy2(src,dst);source_hashes[name]=sha(dst)
    cfg=json.loads((ROOT/'bioclip_lpft/config.json').read_text())
    assert cfg['script_hash']==source_hashes['plant_bioclip_lpft.py']
    versions={p:importlib.metadata.version(p) for p in ['torch','torchvision','open_clip_torch','timm','numpy','scikit-learn','scipy','joblib','threadpoolctl','safetensors','Pillow','huggingface_hub']}
    artifacts={}
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and (p.suffix in ('.pth','.npz','.csv','.joblib') or p.name in ('config.json','cv_report.json','selection_report.json')):
            artifacts[str(p.relative_to(ROOT))]=dict(bytes=p.stat().st_size,sha256=sha(p))
    external=[OLD/'dino_lpft/probabilities.npz',OLD/'dino_lpft/config.json',OLD/'dino_foreground/features.npz',OLD/'dino_foreground/feature_metadata.json']
    external += list(sorted((OLD/'dino_lpft').glob('fold*.pth')))
    external_hashes={str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in external}
    manifest=dict(python=platform.python_version(),packages=versions,sources=source_hashes,artifacts=artifacts,external_sources=external_hashes,
                  official_model=json.loads(Path('work/bioclip25/download_verified.json').read_text()),
                  original_data_hash=cfg['data_hash'],baseline_csv_sha256=sha(OLD/'submission_v3_20260929.csv'))
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (ROOT/'requirements-observed.txt').write_text('\n'.join(f'{p}=={v}' for p,v in versions.items())+'\n')
    environment=json.loads((ROOT/'run_environment.json').read_text())
    environment['bio_lpft_status']='complete'
    environment['source_sha256']=source_hashes
    (ROOT/'run_environment.json').write_text(json.dumps(environment,indent=2))
    text=f'''# 植物分类 v4 实验结果

## 提交文件

- 文件：`submission_v4_20260929.csv`，378 行，列为 `ID,Category`，ID 顺序与原始示例一致。
- 选择方案：**{names[selected['name']]}**（`{selected['name']}`），权重 `{json.dumps(selected['weights'])}`。
- SHA-256：`{report['submission_sha256']}`。
- 与 v3 相比，测试集预测改变 **{report['changed_test_predictions']} / 378** 张。
- v3 的 **0.9735** 来自用户线上反馈；新文件尚无线上分数。本次离线结果不能保证突破 0.9735。

## 同折比较

原 v3 为 0.960（480/500）；选中候选为 **{selected['accuracy']:.3f}（{round(selected['accuracy']*500)}/500）**。
相对原模型，OOF 纠正 {paired['corrected']} 张，新增错分 {paired['newly_wrong']} 张，仍同时错分 {paired['both_wrong']} 张。

{chr(10).join(rows)}

{chr(10).join(by_class)}

所有候选使用相同 seed=42 分层五折。冻结特征的线性和核分类头用内层折选择超参数，图传播参数固定，微调用预先固定末轮 EMA。外层每张验证图未参加对应模型监督训练。
但本轮及前几轮反复使用了这些开发折，且最终从 {len(report['candidates'])} 个候选中选优，因此最优 OOF 成绩存在选择偏差；不能当作独立泛化估计或线上成绩。500 张样本上每多判对一张仅增加 0.002。

## 实际完成的工作

1. 原 DINOv2-L 微调模型的 518 分辨率、八视角推理，以及与 336 输入的等权平均。
2. 固定 HSV/ExG 前景框裁剪；225 张图重新提取特征，653 张经像素一致性核对后复用。完成嵌套五折分类。
3. 官方 BioCLIP 2.5 Huge 图像编码器，完整权重 SHA 校验；提取 878 张图的八视角特征，完成嵌套五折线性分类。
4. BioCLIP 最后两个 Transformer block、末端 LayerNorm 和分类头微调：五折均完成 15 轮；每折最终八视角推理，保存五个实际权重。
5. DINOv2/BioCLIP 特征的 RBF SVM 对照，内层选核尺度、C 和核混合比例。
6. 固定 5 近邻、温度 0.1、alpha=0.2/0.5 的图传播对照；验证/测试仅提供无标签查询特征，监督标签只来自训练子集。其准确率分别为 0.958/0.944，未超过原模型。

BioCLIP 微调折准确率：{[e['final_accuracy'] for e in finals]}。各折训练加末轮推理耗时（秒）：{[e['elapsed'] for e in finals]}。
冻结 BioCLIP 提取实际用时约 319 秒，PyTorch 峰值分配约 1.24 GiB。微调期间一次 nvidia-smi 观测约 4.2 GiB，后者是瞬时占用，不是全程峰值。设备为 RTX 4060 Laptop 8GB。

## 论文、代码与复现

- [BioCLIP 2 论文](https://arxiv.org/abs/2505.23883)与[官方仓库](https://github.com/Imageomics/bioclip-2)：生物图像表征的研究依据。
- [BioCLIP 2.5 Huge 官方模型卡](https://huggingface.co/imageomics/bioclip-2.5-vith14)：本次实际使用的后续 ViT-H/14 模型，区别于 2025 年论文的原模型。
- [LP-FT 论文](https://arxiv.org/abs/2202.10054)：先线性探测再微调的依据。本实现仅微调最后两层，是任务适配。

具体预处理、参数、固定候选、复现命令见 `RESEARCH.md`。代码快照见 `source_snapshot/`，环境版本、源代码、数据指纹、权重和提交哈希见 `manifest.json`。
主要训练代码是 `outputs/plant_bioclip_lpft.py`，复用 `outputs/plant_finetune.py`；候选选择为 `outputs/select_plant_v4.py`。
逐张测试预测变动见 `prediction_changes.csv`；验证集的真值、折号及新旧错分见 `oof_predictions.csv`。所有测试预测由模型概率产生，没有人工改写测试标签。
'''
    (ROOT/'RESULTS.md').write_text(text,encoding='utf-8')
    print('Report and hash manifest complete',flush=True)

if __name__=='__main__':main()
