"""Evidence-based V4 result report and an allowlisted complete project package."""
from datetime import datetime,timedelta,timezone
from pathlib import Path
import hashlib,json,zipfile
import pipeline as base

ROOT=base.ROOT;RUN=ROOT/'v4'


def main():
    m=json.loads((RUN/'metrics.json').read_text(encoding='utf-8'))
    valid=json.loads((RUN/'delivery/submission_validation.json').read_text(encoding='utf-8'))
    assert valid['valid'] and valid['rows']==20
    seed=json.loads((RUN/'seed_check.json').read_text(encoding='utf-8')) if (RUN/'seed_check.json').exists() else None
    summaries=[]
    for f in range(5):
        s=json.loads((RUN/'transfer'/f'fold{f}'/'summary.json').read_text(encoding='utf-8'))
        assert s['trained_step']==2000
        summaries.append(s)
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S +08:00')
    lines=['# 任务4：论文策略后的完整实验与V4提交', '',f'生成时间：{stamp}。','',
           '| 版本 | 五折OOF Dice | Kaggle线上Dice |','|---|---:|---:|',
           '| V1 | 0.837242 | 0.85042（用户反馈） |',
           f"| V2 | {m['baseline_v2_oof']:.6f} | 未提供 |",
           f"| 原生迁移模型单独预测 | {m['new_family_oof']['dice']:.6f} | 尚未提交 |",
           f"| V4最终组合 | {m['selected']['dice']:.6f} | 尚未提交 |",'',
           f"相对V2的五折验证变化为 **{m['delta_vs_v2']:+.6f} Dice**（{m['delta_vs_v2']*100:+.3f}个百分点），{m['images_improved']}/20张图改善。新模型占比{m['selected']['alpha']:.2f}，旧模型占比{1-m['selected']['alpha']:.2f}；统一阈值{m['selected']['threshold']:.3f}。",'',
           '这些是用于选择权重、阈值和组合比例的验证分数，不是完全独立的测试成绩，也不是新的线上成绩。不能将验证提升直接加到0.85042上。','',
           '## 为什么这次改变主路线','',
           '旧模型的误差审计显示：相对最近邻标签，官方脚本复现目标的血管面积增加约49%；66.5%的漏检在新增边界上，80.9%的误检距目标边界不超过2像素，同时细血管中心线召回仅约80.6%。因此不再只做续训和TTA，而是分离图像中血管的学习与提交时的确定性转换。','',
           '最终模型使用torchvision官方ImageNet ResNet-18编码器，另设未下采样的32通道细节分支，再用逐级解码器融合。编码器仅看按ImageNet均值/标准差归一化的RGB；细节分支保留原四通道。编码器BN运行统计冻结，仿射参数可训练；编码器学习率为解码器的0.1倍。','',
           '在原始584×565图像和原始二值血管标签上训练，不先把标签扩成512目标。预测先裁掉补边，再按相同双线性缩放＋非零判定算子映射到512提交网格。为与旧版512输出组合，在100个预定义均匀阈值上执行二值转换后取平均，得到单调置信图；它不是医学上校准的概率。','',
           '使用原来的按图像划分的五折，每折16张训练、4张验证。每折2,000步，batch=8，160×160随机裁块，D4几何与原有颜色增强，AdamW；解码器学习率从0.001余弦衰减至0.00002，前50步预热。每200步按固定原生阈值0.5转换后的验证Dice保存最佳权重；最终使用四种旋转平均。','',
           '官方ImageNet权重的SHA-256经过核对。没有使用公开眼底分割权重或外部眼底标签；这是一种包含通用ImageNet先验的迁移方案，不能称作从零训练。','',
           '## 论文阅读、对照与否决','',
           '完整依据及预设通过条件见 `research-evaluations/论文策略与验证协议.md`。本轮筛选了DSCNet（ICCV 2023）、clDice（CVPR 2021）、GraphSeg（NeurIPS 2025）、FR-UNet（JBHI 2022）和SAUNA（WACV 2025）。最终实现为任务特定的迁移网络，不声称完整复现其中某篇论文，也不把不同数据协议下的文献指标与本比赛直接比较。','',
           'FR-UNet同折1,200步对照：直接扩大目标0.80580、核心辅助0.80839、原生网格0.81408，未达到扩跑标准，因此没有给这些较弱方案扩跑五折。预训练原生路线在两个折的完整预算验证和置信图组合后，平均约0.84389，相比V2相同两折约0.8353提升约0.0086，达到预设0.008门槛，才扩展其余三折。','',
           '整图与重叠裁块推理也做了检查，未见重叠推理收益，因此最终沿用整图推理。官方FR-UNet本身也使用整图测试，不能将其当作实现错误。没有将骨架、软标签、动态蛇形卷积和图先验一次堆叠到最终模型里。','',
           '## 五折最终组合结果','', '| 折 | V2 | V4 | 变化 |','|---|---:|---:|---:|']
    for r in m['fold_comparison']:lines.append(f"| {r['fold']} | {r['baseline_v2']:.6f} | {r['v4']:.6f} | {r['delta']:+.6f} |")
    if seed:
        lines.extend(['','## 独立随机种子复核','',
                      f"难折2：V2为{seed['baseline_v2']:.6f}，主种子最终组合为{seed['main_seed']:.6f}，额外种子为{seed['alternate_seed']:.6f}。额外种子使用完全相同的全局阈值和组合比例，没有重新调参；相对V2变化为{seed['alternate_delta_vs_v2']:+.6f}。",'',
                      '该复核只是一折上的额外种子诊断，不等同于五折多种子置信区间。额外种子权重未加入测试提交。'])
    lines.extend(['','## 误差变化的边界','',
                  '在同一组用于模型选择的OOF图像上，新版新增边界漏检从76,118降到68,364，误检从99,597降到97,693；但核心漏检从38,279增加到39,386，细血管中心线召回从80.63%降到79.57%。因此本次收益主要体现为提交网格边界与整体重叠改善，不能宣称细血管或拓扑连接已经改善。这是条件诊断，不是独立因果消融。详见mechanism_audit.json。',
                  '','## 提交与复现','',
                  '`v4/delivery/submission_v4.csv` 是本次建议测试的文件。20行，Id/Predicted列，按样例顺序排列，白色为血管前景。官方脚本原封不动生成RLE，再由独立编码器/解码器逐像素核对，包括正数、排序、范围和重叠。另附黑血管、白背景图片。','',
                  f"CSV SHA-256：`{valid['sha256']}`。",'',
                  '在task4目录使用交付权重重新推理：','', '```powershell','python -u -X utf8 predict_v4.py','```','',
                  '只用保存的512置信图重新导出（无需重新运行GPU模型）：','', '```powershell','python -u -X utf8 predict_v4.py --cached','```','',
                  '从已核对的通用ImageNet初始化完整训练：','', '```powershell',
                  'python -u -X utf8 v4_transfer.py --folds 0 1 2 3 4 --steps 2000 --stop-at 2000',
                  'python -u -X utf8 predict_v4.py','```','',
                  '重新训练会覆盖相应输出，先保留交付模型副本。包内没有未使用实验的优化器大快照；从零重训从ImageNet初始化开始。固定种子不能保证不同CUDA软硬件上的逐位相同训练结果，交付权重的推理另做解压后字节一致性复核。','',
                  '五折新模型、旧组合依赖、原始数据、通用初始化、论文源代码许可证、训练参数、逐轮日志与审计文件均在完整包中。V1/V2原提交保留。','',
                  '## 逐图结果','', '| ID | V2 | V4 | 变化 |','|---|---:|---:|---:|'])
    for r in m['per_image']:lines.append(f"| {r['id']} | {r['baseline_v2']:.6f} | {r['v4']:.6f} | {r['delta']:+.6f} |")
    (RUN/'结果与复现说明.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    topfiles=['pipeline.py','improve.py','predict_deterministic.py','predict_v2.py','postprocess_probe.py',
              'v3.py','v3_native.py','v4_transfer.py','predict_v4.py','operator_confidence.py','verify_v4_seed.py',
              'make_v4_delivery.py','requirements.txt','verify_contract.py','README_中文.md','audit.json','data_manifest.json']
    files={ROOT/name for name in topfiles}
    for directory in ('data','third_party','research-evaluations'):
        files.update(p for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.npz')
    files.update([ROOT/'pretrained/resnet18-f37072fd.pth',ROOT/'pretrained/resnet18_provenance.json'])
    for directory in ('runs','v2','v4'):
        for p in (ROOT/directory).rglob('*'):
            if not p.is_file():continue
            rel=p.relative_to(ROOT/directory)
            if any(s in rel.parts for s in ('official_export','pilots','deps','pip_cache','runtime_temp','source_versions','smoke','verify_v4')):continue
            if p.name=='latest.pt' or p.name=='native_probabilities.npz':continue
            if p.suffix=='.pt' and ('seed10000' in rel.parts or p.name!='best.pt'):continue
            if p.suffix=='.npz' and p.name not in ('probabilities.npz','validation.npz','lifted_confidence.npz','selected_probabilities.npz'):continue
            if p.suffix in ('.log','.json','.md','.npz','.pt','.png','.csv'):files.add(p)
    # Experimental source hashes/configs/history without unused model snapshots.
    files.update(p for p in (ROOT/'v3').rglob('*') if p.is_file() and p.suffix in ('.py','.json','.log'))
    archive=ROOT.parent/'task4_v4_完整交付.zip';contents=[]
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(files):
            assert p.is_file(),p
            name='task4/'+p.relative_to(ROOT).as_posix();z.write(p,name)
            contents.append(dict(path=name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for r in contents:assert hashlib.sha256(z.read(r['path'])).hexdigest()==r['sha256']
    result=dict(archive=str(archive),bytes=archive.stat().st_size,files=len(contents),
                sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),crc_valid=True,contents_verified=True,
                selected=m['selected'],contents=contents)
    (ROOT.parent/'task4_v4_交付校验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='contents'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
