"""Write the evidence-based result report and package the finished project."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent


def main():
    metrics = json.loads((ROOT/'runs/metrics.json').read_text(encoding='utf-8'))
    validation = json.loads((ROOT/'delivery/submission_validation.json').read_text(encoding='utf-8'))
    config = json.loads((ROOT/'runs/config.json').read_text(encoding='utf-8'))
    assert validation['valid'] and validation['rows'] == 20
    assert len(metrics['folds']) == 5
    current_hash = hashlib.sha256((ROOT/'pipeline.py').read_bytes()).hexdigest()
    assert current_hash == config['script_sha256'], 'Pipeline changed after the recorded training configuration'
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S +08:00')
    text = [
        '# 任务4：训练结果与提交说明', '', f'生成时间：{stamp}。', '',
        '五折训练、五模型旋转预测集成和全部20张测试图的CSV生成均已完成。', '',
        f"- 主提交：`delivery/submission.csv`，20行，列名 `Id,Predicted`。",
        f"- 五折OOF平均 Dice，固定阈值0.5：**{metrics['oof_dice_05']:.6f}**。",
        f"- OOF选择阈值：**{metrics['selected_threshold']:.3f}**；选择后平均 Dice：**{metrics['oof_dice_selected']:.6f}**。",
        f"- 最近邻标签在上述阈值下的OOF Dice：**{metrics['nearest_label_dice_at_selected']:.6f}**。",
        '- 未上传Kaggle；没有线上测试分数。', '',
        '主分数使用官方脚本的标签转换规则：原标签双线性缩放至512×512后，任意非零值算血管。该规则扩大血管边缘；不得将该分数当作原始分辨率或其他二值化规则下的Dice。最近邻标签的分数是不同目标定义下的补充诊断。', '',
        '验证图参与了权重与阈值选择，因此以上OOF分数是模型选择阶段的验证估计，不能称为完全独立的测试结果。训练与验证按整张图像分离，未使用测试标签、外部眼底标签或预训练权重。', '',
        '| 折（从0开始） | 验证图ID | 最佳步数 | 不含旋转平均 Dice@0.5 | 含旋转平均 Dice@0.5 |',
        '|---|---|---:|---:|---:|',
    ]
    for f in metrics['folds']:
        text.append(f"| {f['fold']} | {', '.join(f['valid_ids'])} | {f['best_step']} | {f['best_dice_no_tta']:.6f} | {f['dice_tta']:.6f} |")
    text.extend(['', '每折均完成1,800步，最终使用该折验证分数最高的已保存权重。', '',
                 '## 极性和提交格式', '',
                 '实际标签是白血管、黑背景，用户已确认遵循实际标签和官方脚本。主CSV编码血管前景；`delivery/vessel_white`为其对应PNG。`delivery/vessel_black`是题面要求的黑血管、白背景图，二者逐像素互补，使用反色图运行脚本会得到不同语义的CSV。', '',
                 '全部20行已按样例顺序排列，并通过官方脚本生成结果与独立RLE编解码器的逐像素一致性检查，包括正数、起点排序、无重叠、范围、行展开方向及首末像素。所有PNG为512×512无损二值图。', '',
                 f"CSV SHA-256：`{validation['sha256']}`。", '',
                 '## 复现', '',
                 '在工程目录运行 `python -u -X utf8 predict_deterministic.py` 可使用已交付五折权重重新生成提交；运行 `python -X utf8 pipeline.py validate` 校验提交。完整从零训练命令为 `python -u -X utf8 pipeline.py train --steps 1800 --eval-every 150`，会覆盖对应折权重；之后运行确定性推理入口生成最终提交。更多参数与方法详见 `README_中文.md`。', '',
                 '固定种子为20261103。CUDA可能存在微小数值差异；配置、脚本哈希、官方脚本哈希、逐轮日志、数据清单、权重及OOF概率均已保存。', '',
                 '## 逐图OOF分数', '', '| 图像ID | Dice（选择阈值） |', '|---|---:|'])
    for row in metrics['per_image']:
        text.append(f"| {row['id']} | {row['dice']:.6f} |")
    report = '\n'.join(text)+'\n'
    (ROOT/'结果说明.md').write_text(report, encoding='utf-8')
    (ROOT/'delivery/结果说明.md').write_text(report, encoding='utf-8')
    # The archive has a single task4/ root and includes all data and final weights.
    archive = ROOT.parent/'task4_完整交付.zip'
    included = []
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(ROOT.rglob('*')):
            if not p.is_file():
                continue
            rel = p.relative_to(ROOT)
            if any(part in {'__pycache__', 'verification_extract'} for part in rel.parts):
                continue
            if len(rel.parts)>1 and rel.parts[0]=='runs' and rel.parts[1] in {'smoke', 'official_export'}:
                continue
            if p.suffix in {'.tmp', '.zip'}:
                continue
            name = 'task4/'+rel.as_posix()
            z.write(p, name)
            included.append(dict(path=name, bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(included)
        for row in included:
            assert hashlib.sha256(z.read(row['path'])).hexdigest() == row['sha256']
    manifest = dict(archive=str(archive), bytes=archive.stat().st_size, files=len(included),
                    sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), crc_valid=True,
                    contents_sha256_verified=True, included=included)
    (ROOT.parent/'task4_交付包校验.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k!='included'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
