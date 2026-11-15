"""Verify final artifacts and package source/logs without putting model weights on C:."""
import csv,hashlib,json,zipfile,shutil
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; OUT=HERE.parent


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def main():
    with open(r'D:\下载\submission.csv',encoding='utf-8-sig',newline='') as f: template=list(csv.DictReader(f))
    with (OUT/'submission.csv').open(encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f);columns=reader.fieldnames;rows=list(reader)
    assert columns==['ID','Emotion']
    assert [r['ID'] for r in rows]==[r['ID'] for r in template]
    labels=np.array([int(r['Emotion']) for r in rows]);assert set(labels)<=set(range(6))
    assert len(rows)==7065 and len(set(r['ID'] for r in rows))==7065
    selection=json.loads((OUT/'model_selection.json').read_text(encoding='utf-8'))
    assert selection['completed']
    probabilities=np.load(OUT/'final_probabilities.npz')
    assert np.array_equal(probabilities['ids'],np.array([r['ID'] for r in rows]))
    assert probabilities['probabilities'].shape==(7065,6)
    assert np.isfinite(probabilities['probabilities']).all()
    assert np.allclose(probabilities['probabilities'].sum(1),1,atol=1e-5)
    assert np.array_equal(labels,probabilities['probabilities'].argmax(1))
    verification={'rows':len(rows),'columns':columns,'id_order_matches_template':True,
                  'label_counts':np.bincount(labels,minlength=6).tolist(),
                  'submission_sha256':hashlib.sha256((OUT/'submission.csv').read_bytes()).hexdigest(),
                  'seed':42,'model_checkpoint_paths':[str(Path(r'D:\FER2026task2\runs')/name/'best.pt') for name in selection['final_runs']]}
    if selection.get('upgrade',{}).get('adopted_over_core'):
        verification['model_checkpoint_paths']=selection['upgrade']['final_checkpoint_paths']
        verification['upgrade_selected']=selection['upgrade']['selected']
        verification['source_image_overlap_possible']=selection['upgrade']['source_image_overlap_possible']
    if 'postprocessing' in selection:
        verification['core_postprocessing']=selection['postprocessing']['selected']
    if selection.get('dima_finetune',{}).get('completed'):
        ft=selection['dima_finetune']
        verification['model_checkpoint_paths']=ft['final_checkpoint_paths']
        verification['dima_selected']=ft['selected']
        verification['source_image_overlap_possible']=True
        if 'dima_tta' in selection:
            verification['dima_test_time_adaptation']=selection['dima_tta']['selected_config']
    verification['checkpoints']=[{'path':path,'sha256':file_sha(path)}
                                 for path in verification['model_checkpoint_paths']]
    (OUT/'submission_verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
    if selection.get('dima_finetune',{}).get('completed'):
        dest=OUT/'run_logs'/'dima';dest.mkdir(parents=True,exist_ok=True)
        for run in Path(r'D:\FER2026task2\runs').glob('dima_ft_*_s42'):
            folder=dest/run.name;folder.mkdir(exist_ok=True)
            for name in ['config.json','history.json','completed.json']:
                if (run/name).exists():shutil.copy2(run/name,folder/name)
        for name in ['dima_finetune.log','dima_finetune.err','dima_tta.log','dima_tta.err']:
            src=Path(r'D:\FER2026task2\runs')/name
            if src.exists():shutil.copy2(src,dest/name)
    lines=['# 六类人脸表情分类实验结果','',
           '本文件由实际完成的运行日志生成。指标来自一次固定的训练集留出验证，不是 Kaggle 排名或测试分数。',
           '', '## 实验设置','',
           '- 唯一种子：42；不做 K 折或多种子训练。',
           '- 训练子集 24,019 张，验证 4,256 张；相同像素重复图按组划分。',
           '- 按验证选定轮数，用全部 28,275 张有标签图重训/适配；零轮最优则保留原始权重。测试集 7,065 张无标签。',
           '- 原始标签未修改；所有权重、特征和模型缓存保存在 D 盘。',
           '', '## 已完成的基础方案比较','',
           '| 方案 | 最佳轮数 | 翻转 TTA 准确率 | Macro-F1 |',
           '|---|---:|---:|---:|']
    if selection.get('dima_finetune',{}).get('completed'):
        ft=selection['dima_finetune']
        lines[2:2]=[f"**最终提交采用 Dima：{ft['selected']['mode']}，固定留出验证准确率 {selection['final_validation_accuracy']:.2%}；已完成全量重训。**",'']
    for r in selection['results']:
        lines.append(f"| {r['variant']} | {r['best_epoch']} | {r['accuracy']:.2%} | {r['macro_f1']:.2%} |")
    lines.extend(['',f"B0 历史阶段选择的架构：{', '.join(selection['selected_variants'])}。最优两模型等权融合的验证准确率为 {selection['top2_ensemble_validation_accuracy']:.2%}；只有比最佳单模型高至少 0.25 个百分点时才采用。",''])
    if 'postprocessing' in selection:
        post=selection['postprocessing']; best=post['selected']; base=post['baseline']
        lines.extend(['## TTA 与深度特征优化','',
                      '| TTA | 近邻权重 | 准确率 | Macro-F1 |', '|---|---:|---:|---:|'])
        for r in post['candidates']:
            lines.append(f"| {r['tta']} | {r['knn_weight']:.1f} | {r['accuracy']:.2%} | {r['macro_f1']:.2%} |")
        lines.extend(['',f"最终设置：**{best['tta']}，近邻权重 {best['knn_weight']:.1f}**。验证准确率 **{best['accuracy']:.2%}**，Macro-F1 **{best['macro_f1']:.2%}**。相对已选模型的原始翻转 TTA，准确率变化 {(best['accuracy']-base['accuracy'])*100:+.2f} 个百分点。",'',
                      f"验证中纠正原错误 {post['validation_corrected']} 张，同时新增错误 {post['validation_new_errors']} 张。TTA 和近邻融合每步均要求至少 0.25 个百分点增益；这不是统计显著性检验。",'',
                      '单次验证被用于选择模型、checkpoint 和推理设置，分数有选择偏差。全量训练模型没有另外可用的独立带标签测试评估。',''])
    error_file=OUT/'error_analysis'/'summary.json'
    if error_file.exists():
        e=json.loads(error_file.read_text(encoding='utf-8'))
        lines.extend(['## 错误分析','',f"最终验证误分类 {e['validation_error_count']} 张；训练复查候选 {e['training_review_candidates']} 张；同像素但标签冲突 {e['duplicate_label_conflict_groups']} 组。候选不等于确定错标，未自动删除或改标签。",'',
                      '见 `error_analysis/error_gallery.html` 的误分类及三个训练近邻，`validation_errors.csv` 的全量错误表，以及 `training_review_candidates.csv` 的训练复查列表。',''])
    if selection.get('upgrade',{}).get('adopted_over_core'):
        upgrade=selection['upgrade'];m=upgrade['selected_metrics']
        lines.extend(['## 最终采用的预训练升级模型','',
                      f"上文为 B0 阶段记录。最终提交已替换为预训练升级方案，所选验证准确率 **{m['accuracy']:.2%}**，Macro-F1 **{m['macro_f1']:.2%}**。",'',
                      '具体模型、特征适配和翻转设置见 `预训练升级结果.md`。原 error_analysis 属于 B0 阶段，最终模型错误见 `upgrade_validation_errors.csv`。',
                      'FERPlus/FER2013 权重可能包含本题源图像，必须与独立泛化结果区分。这不是 Kaggle 实测分数。',''])
    if selection.get('dima_finetune',{}).get('completed'):
        ft=selection['dima_finetune'];s=ft['selected']
        lines.extend(['## Dima 主干微调与论文方法','',
                      '最终结果以本节为准。单种子 42，原有分组留出；解冻 ViT 最后四个块与最终归一化、分类头。比较双视图 CE 与 CE+SupCon+概率一致性，后者借鉴 NeurIPS 2020 SupCon 和 AAAI 2025 NLA 的思想，并非完整复现 NLA。','',
                      '|方法|最优轮次|TTA|验证准确率|Macro-F1|','|---|---:|---|---:|---:|',
                      '|原始 Dima|0|single|81.98%|81.26%|','|冻结主干线性适配|13|flip|82.35%|81.61%|'])
        for r in ft['experiments']:lines.append(f"|{r['mode']} 主干微调|{r['epoch']}|{r['tta']}|{r['accuracy']:.2%}|{r['macro_f1']:.2%}|")
        lines.extend(['',f"最终采用 **{s['mode']}，{s['epoch']} 轮，{s['tta']}**；验证准确率 **{ft['selected_metrics']['accuracy']:.2%}**。完成全量训练后生成提交文件。",'',
                      '保留既有最佳模型；新方案至少提升 0.25 个百分点才替换。这里没有保证论文损失一定有效。预训练源图像重叠风险已向用户披露并获得继续使用授权，成绩不能作为无污染独立泛化评估。',
                      '复现最终预测：`python fer_project/infer_dima_finetuned.py`。微调仅保存可训练参数差量，必须结合固定版本的 Dima 原始权重加载。',''])
    if 'dima_tta' in selection:
        tta=selection['dima_tta']
        lines+=['## 推理时微调','',
                 '比较 EATA 启发的可靠样本熵最小化和 MEMO 启发的多视图边际熵，均只更新末端 LayerNorm。先以固定种子打乱图像再分批，每批无标签适配两步并重置参数；不是完整复现原论文。','',
                 '|方法|学习率|验证准确率|Macro-F1|','|---|---:|---:|---:|']
        for r in tta['candidates']:lines.append(f"|{r['config']['name']}|{r['config']['lr']}|{r['metrics']['accuracy']:.2%}|{r['metrics']['macro_f1']:.2%}|")
        lines+=['',f"最终是否采用推理微调：**{tta['adopted']}**；最终验证准确率 **{tta['selected_metrics']['accuracy']:.2%}**。至少提高 0.25 个百分点才替换。",'',
                 '相关论文：[MEMO（NeurIPS 2022）](https://github.com/zhangmarvin/memo)、[EATA（ICML 2022）](https://github.com/mr-eggplant/EATA)。对照配置与重置验证见 dima_tta_results.json、dima_tta_verification.json。',
                 '适配对照中的不适配模型以 batch=16、打乱顺序预测，准确率为 83.18%；与标准 batch=32 的混合精度预测相差一张。正式基准仍保留 83.15%，所有适配配置均未超过两种基准。','']
    if (OUT/'dima_error_comparison.json').exists():
        errors=json.loads((OUT/'dima_error_comparison.json').read_text())
        lines+=['## 最终模型错误与特征诊断','',
                 f"相对冻结主干线性适配，纠正 {errors['corrected_errors']} 张误判、新增 {errors['new_errors']} 张误判。逐类召回与混淆变化见 [Dima错误分析.md](Dima错误分析.md)。",'',
                 '固定抽样的 CLS 特征诊断未显示 SupCon 比 CE 有额外的类别分离优势，不能据此宣称对比损失改善了特征结构。抽样和数值见 [Dima特征诊断.md](Dima特征诊断.md)。','']
    lines.extend(['## 提交与复现','',
                  f"`submission.csv` 已校验 {len(rows)} 行、列名 ID/Emotion、ID 顺序与原始模板一致、类别范围 0–5、标签与保存概率的 argmax 一致。未自动提交 Kaggle。",'',
                  '标签：0 愤怒；1 恐惧；2 快乐；3 悲伤；4 惊讶；5 中性。', '',
                  '复现流程与论文/作者代码来源见 `fer_project/README.md`；checkpoint 路径与 SHA256 见 `submission_verification.json`。模型权重没有放进 C 盘的代码 ZIP。', '',
                  'B0 历史代码包含 EfficientFace（AAAI 2021）、SCN（CVPR 2020）及 RUL（NeurIPS 2021）思想改造；RUL 未完成运行，不计入实测结果。最终主线为 Dima 主干微调及上面的 SupCon/MEMO/EATA 对照。各方法是否有效，以实际结果为准。'])
    (OUT/'实验结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    warning_file=OUT/'pipeline_warnings.json'
    if warning_file.exists():
        warnings=json.loads(warning_file.read_text(encoding='utf-8'))
        if warnings:
            with (OUT/'实验结果.md').open('a',encoding='utf-8') as f:
                f.write('\n## 未完成阶段\n\n'+json.dumps(warnings,ensure_ascii=False,indent=2)+'\n')
    archive=OUT/'FER2026_source_and_results.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in HERE.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and (path.suffix in ['.py','.md','.txt','.json'] or path.name=='LICENSE'):
                z.write(path,'fer_project/'+path.relative_to(HERE).as_posix())
        for name in ['submission.csv','submission_verification.json','experiment_results.csv','experiment_results.json',
                     'model_selection.json','data_audit.json','sample_grid.jpg','module_verification.json',
                     'pretrained_manifest.json','validation_split.json','optimization_results.csv',
                     'optimization_results.json','postprocessing_verification.json','submission_core.csv','model_selection_core.json','pipeline_warnings.json','实验结果.md']:
            if (OUT/name).exists(): z.write(OUT/name,name)
        for path in OUT.iterdir():
            if path.is_file() and (path.name.startswith(('upgrade_','hf_','dima_')) or path.name in ['论文检索与升级方案.md','预训练升级结果.md','HF模型筛选.md','数据泄露审计.md','Dima主干微调方案.md','Dima错误分析.md','Dima特征诊断.md','推理时适配方案.md','leakage_audit.json']):
                z.write(path,path.name)
        if (OUT/'run_logs').exists():
            for path in (OUT/'run_logs').rglob('*'):
                if path.is_file(): z.write(path,path.relative_to(OUT))
        if (OUT/'error_analysis').exists():
            for path in (OUT/'error_analysis').rglob('*'):
                if path.is_file(): z.write(path,path.relative_to(OUT))
    with zipfile.ZipFile(archive) as z: assert z.testzip() is None
    print(json.dumps(verification),flush=True)

if __name__=='__main__': main()
