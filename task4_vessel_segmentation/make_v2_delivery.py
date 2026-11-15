"""V2 evidence report and complete reproducible archive, leaving V1 intact."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, ImageDraw
import improve as imp
from postprocess_probe import refine

ROOT, V2 = imp.ROOT, imp.V2


def main():
    m = json.loads((V2/'metrics.json').read_text(encoding='utf-8'))
    check = json.loads((V2/'delivery/submission_validation.json').read_text(encoding='utf-8'))
    cfg = json.loads((V2/'train_config.json').read_text(encoding='utf-8'))
    assert check['valid'] and check['rows']==20
    assert cfg['script_sha256']==hashlib.sha256((ROOT/'improve.py').read_bytes()).hexdigest()
    scores = []
    for fold in range(5):
        s = json.loads((V2/'runs/control'/f'fold{fold}'/'summary.json').read_text(encoding='utf-8'))
        assert not set(s['train_ids'])&set(s['valid_ids'])
        assert s['steps']==900
        scores.append(s)
    pilot = json.loads((V2/'pilot_summary.json').read_text(encoding='utf-8'))
    diagnostic = json.loads((V2/'diagnostic.json').read_text(encoding='utf-8'))
    delta = m['selected']['dice']-m['baseline_oof']['dice']
    wins = sum(r['delta']>0 for r in m['per_image'])
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S +08:00')
    lines = ['# 任务4 V2：提升实验与提交说明', '', f'生成时间：{stamp}。', '',
             'V1线上分数 **0.85042** 来自用户反馈。V2尚未上传Kaggle，不能将下面的验证提升称作已经获得的线上提升。', '',
             '| 结果 | 五折OOF平均Dice | Kaggle线上Dice |', '|---|---:|---:|',
             f"| V1（旧模型） | {m['baseline_oof']['dice']:.6f} | 0.85042（用户反馈） |",
             f"| V2继续训练模型＋镜像预测 | {m['variant_oof']['dice']:.6f} | 尚未提交 |",
             f"| V2最终选定组合 | {m['selected']['dice']:.6f} | 尚未提交 |", '',
             f"验证变化为 **{delta:+.6f} Dice**（{delta*100:+.3f}个百分点）；{wins}/20张图改善，{20-wins}/20张持平或回退。最终新模型权重占比 {m['selected']['alpha']:.2f}，原模型权重占比 {1-m['selected']['alpha']:.2f}；统一阈值 {m['selected']['threshold']:.3f}。", '',
             '指标与旧版使用完全相同的目标定义：原始白血管标签双线性缩放至512×512后，任意非零值作为血管。仍按整张图像的原五折划分训练和验证。验证图没有进入对应模型的训练；验证结果用于选择权重、阈值及集成比例，所以这属于模型选择阶段的验证估计。', '',
             '## 采用的改动', '',
             '从V1对应折的最佳权重继续训练，每折追加900次更新，batch=12，裁块160×160。网络和原始四通道输入保持一致。AdamW，学习率从0.0002余弦衰减到0.00002；使用衰减0.98的指数移动参数平均（EMA），每100步按验证Dice@0.5保存最佳EMA权重，并保留未更新的起点作为候选。增强方式沿用原方案。', '',
             '推理增加镜像，与0/90/180/270度旋转共同构成8种变换平均。五折模型各自预测20张测试图，再平均；与冻结的V1预测在0/0.25/0.5/0.75/1.0五个权重中，仅使用OOF结果选择组合。阈值范围0.3至0.7，步长0.025。测试标签和新的线上反馈均未参与选择。', '',
             f"孤立区域过滤：{'采用' if m['postprocess']['adopted'] else '未采用'}，最小区域为{m['postprocess']['minimum_component_size']}像素（8连通）。该固定尺寸先在V1的OOF诊断中选定，再要求新版中至少12张图、至少4折有正收益，且平均提升大于0.0002才启用；仍属于验证集上的选择。", '',
             '## 小实验与未采用的改动', '',
             '| 现有模型的推理方式 | OOF Dice |', '|---|---:|',
             f"| 4种旋转（V1） | {diagnostic['baseline_d4']['dice']:.6f} |",
             f"| 8种旋转/镜像 | {diagnostic['d8']['dice']:.6f} |",
             f"| 三尺度＋8种变换 | {diagnostic['ms_d8']['dice']:.6f} |", '',
             '多尺度未改善，未用于最终提交。绿色通道CLAHE＋224裁块在折0和折2各做300步初测，没有超过相同步数的原结构继续训练，未扩跑。CLAHE实现遵循局部直方图裁剪、均衡和插值原理，使用NumPy实现，不宣称与OpenCV逐位一致；方法说明：[OpenCV CLAHE](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html)。仅用训练图重新估计BatchNorm统计量也降低了OOF分数，未使用。', '',
             '| 初测折 | 方案 | 固定阈值0.5的最佳Dice |', '|---|---|---:|']
    for r in pilot:
        lines.append(f"| {r['fold']} | {r['variant']} | {r['best']:.6f} |")
    lines.extend(['', '## 五折正式继续训练', '', '| 折 | 验证图 | 起点Dice@0.5 | 最佳EMA Dice@0.5 | 最佳步数 |', '|---|---|---:|---:|---:|'])
    for r in scores:
        lines.append(f"| {r['fold']} | {', '.join(r['valid_ids'])} | {r['initial']:.6f} | {r['best']:.6f} | {r['best_step']} |")
    lines.extend(['', '上表均为4种旋转预测；最终组合表采用8种变换，不能把两表差异全部归因于继续训练。', '',
                  '## 提交与复现', '',
                  '`v2/delivery/submission_v2.csv` 是本次建议提交的文件。20行按原sample_submission顺序排列，编码白色血管前景。原封不动运行官方脚本，再用独立编码器和解码器逐像素核对。`vessel_white` 与CSV一致；`vessel_black`是文字说明中的黑血管图片。', '',
                  f"CSV SHA-256：`{check['sha256']}`。", '',
                  '在task4目录运行：', '', '```powershell',
                  'python -u -X utf8 predict_v2.py',
                  '```', '',
                  '完整复现追加训练：', '', '```powershell',
                  'python -u -X utf8 improve.py train --folds 0 1 2 3 4 --variants control --steps 900',
                  'python -u -X utf8 predict_v2.py',
                  '```', '',
                  '需要旧版runs中的五折起点权重、旧版probabilities.npz及原数据，完整交付包均包含这些文件。旧版pipeline.py、权重、CSV和交付ZIP保留。训练仍可能因CUDA数值差异出现微小变化；推理使用固定算法，并用解压后重跑核对提交。', '',
                  '## 逐图OOF比较', '', '| 图ID | V1 | 最终V2 | 变化 |', '|---|---:|---:|---:|'])
    for r in m['per_image']:
        lines.append(f"| {r['id']} | {r['baseline']:.6f} | {r['v2']:.6f} | {r['delta']:+.6f} |")
    (V2/'提升结果说明.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    # Visual checks include both largest gain and largest regression, not only wins.
    _,y,_,ids=imp.arrays('train')
    stored=np.load(V2/'probabilities.npz')
    old=np.load(ROOT/'runs/probabilities.npz')['oof'].astype(np.float32)
    new=stored['oof']
    order=np.argsort([r['delta'] for r in m['per_image']])
    selected=[int(order[0]),int(order[1]),int(order[-2]),int(order[-1])]
    sheet=Image.new('RGB',(1024,1024),'black')
    draw=ImageDraw.Draw(sheet)
    for row,i in enumerate(selected):
        orig=Image.open(imp.base.DATA/'train/image'/f'{ids[i]}.jpg').convert('RGB')
        a=(old[i]>=.475).astype(np.uint8)*255
        b=refine(new[i:i+1],m['selected']['threshold'],minimum=m['postprocess']['minimum_component_size'])[0].astype(np.uint8)*255
        images=[orig,Image.fromarray((y[i,0]*255).astype(np.uint8)),Image.fromarray(a),Image.fromarray(b)]
        for col,im in enumerate(images):
            sheet.paste(im.convert('RGB').resize((256,256)),(256*col,256*row))
        draw.text((3,row*256+3),f"OOF id={ids[i]} delta={m['per_image'][i]['delta']:+.4f}",fill='white')
    sheet.save(V2/'delivery/oof_comparison.png')
    preview=Image.new('RGB',(768,1024),'black')
    for row,i in enumerate([0,4,9,19]):
        name=str(stored['test_ids'][i])
        orig=Image.open(imp.base.DATA/'test/image'/f'{name}.jpg').convert('RGB')
        old_mask=Image.open(ROOT/'delivery/vessel_white'/f'{name}.png')
        new_mask=Image.open(V2/'delivery/vessel_white'/f'{name}.png')
        for col,im in enumerate([orig,old_mask,new_mask]):
            preview.paste(im.convert('RGB').resize((256,256)),(col*256,row*256))
    preview.save(V2/'delivery/test_comparison.png')
    archive=ROOT.parent/'task4_v2_完整交付.zip'
    contents=[]
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(ROOT.rglob('*')):
            if not p.is_file():
                continue
            rel=p.relative_to(ROOT)
            if any(k in rel.parts for k in ('__pycache__','verification_extract','verify_v2','deps','pip_cache','runtime_temp','official_export')):
                continue
            if p.suffix in ('.tmp','.zip'):
                continue
            if 'pilots' in rel.parts and p.name in ('best.pt','validation.npz'):
                continue
            if len(rel.parts)>1 and rel.parts[0]=='runs' and rel.parts[1]=='smoke':
                continue
            name='task4/'+rel.as_posix()
            z.write(p,name)
            contents.append(dict(path=name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for r in contents:
            assert hashlib.sha256(z.read(r['path'])).hexdigest()==r['sha256']
    summary=dict(archive=str(archive),bytes=archive.stat().st_size,files=len(contents),
                 sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),crc_valid=True,contents_verified=True,
                 oof_delta=delta,selected=m['selected'],contents=contents)
    (ROOT.parent/'task4_v2_交付校验.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='contents'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
