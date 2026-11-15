"""Predict test images using the validation-selected open-model/V4 mixture."""
import argparse,csv,hashlib,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import evaluate_open_rlad as r
from operator_confidence import lift

ROOT=r.ROOT
OUT=ROOT/'open_models/delivery'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cached',action='store_true');args=ap.parse_args()
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    m=json.loads((ROOT/'open_models/rlad_evaluation_unpadded/metrics.json').read_text(encoding='utf-8'))
    selected=m['calibrated_combination']
    assert m['geometry']=='unpad' and selected['dice']>m['baseline_v4']
    old=np.load(ROOT/'v4/selected_probabilities.npz')
    ids=old['test_ids'].tolist()
    OUT.mkdir(exist_ok=True)
    cache=OUT/'test_confidence.npz'
    if args.cached:
        z=np.load(cache);assert z['ids'].tolist()==ids;confidence=z['confidence']
    else:
        assert hashlib.file_digest((r.MODEL/'model.safetensors').open('rb'),'sha256').hexdigest()==m['weights_sha256']
        model=r.load_model();processor=r.Swinv2SegmenterImageProcessor();predictions=[]
        with torch.inference_mode():
            for name in ids:
                im=Image.open(r.base.DATA/'test/image'/f'{name}.jpg').convert('RGB')
                predictions.append(r.predict_one(model,processor,im,'unpad'))
                print('TEST_PREDICTED '+name,flush=True)
        p=np.stack(predictions);assert p.shape==(20,584,565) and np.isfinite(p).all()
        confidence=lift(p)
        np.savez_compressed(cache,ids=np.array(ids),confidence=confidence)
    final=selected['alpha']*confidence+(1-selected['alpha'])*old['test']
    assert final.shape==(20,512,512) and np.isfinite(final).all()
    white=OUT/'vessel_white';black=OUT/'vessel_black'
    white.mkdir(exist_ok=True);black.mkdir(exist_ok=True)
    for name,a in zip(ids,final):
        mask=(a>=selected['threshold']).astype('uint8')*255
        Image.fromarray(mask).save(white/f'{name}.png')
        Image.fromarray(255-mask).save(black/f'{name}.png')
    stage=OUT/'official_export';(stage/'image').mkdir(parents=True,exist_ok=True)
    for p in white.glob('*.png'):shutil.copy2(p,stage/'image'/p.name)
    shutil.copy2(r.base.DATA/'segmentation_to_csv.py',stage/'segmentation_to_csv.py')
    subprocess.run([sys.executable,'-X','utf8',str(stage/'segmentation_to_csv.py')],cwd=stage,check=True)
    with (stage/'submission.csv').open(encoding='utf-8',newline='') as f:
        rows={row['Id']:row['Predicted'] for row in csv.DictReader(f)}
    with (r.base.DATA/'sample_submission.csv').open(encoding='utf-8',newline='') as f:
        order=[row['Id'] for row in csv.DictReader(f)]
    assert len(rows)==20 and set(rows)==set(ids)==set(order)
    dest=OUT/'submission_open_rlad.csv'
    with dest.open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['Id','Predicted'])
        for name in order:
            mask=np.asarray(Image.open(white/f'{name}.png'))>0
            assert rows[name]==r.base.encode(mask)
            assert np.array_equal(r.base.decode(rows[name]),mask)
            assert np.array_equal(np.asarray(Image.open(black/f'{name}.png')),(~mask).astype('uint8')*255)
            assert 0<mask.mean()<.5
            writer.writerow([name,rows[name]])
    validation=dict(valid=True,rows=20,sha256=hashlib.file_digest(dest.open('rb'),'sha256').hexdigest(),
                    selected=selected,baseline_v4_validation=m['baseline_v4'],
                    delta_validation=selected['dice']-m['baseline_v4'],
                    weights_sha256=m['weights_sha256'],geometry='crop known 14px padded input margin at 1472 decoder grid',
                    rule='alpha*open_model_confidence+(1-alpha)*V4_confidence',
                    foreground='white vessels',official_and_independent_RLE_match=True,
                    kaggle_uploaded=False,online_score=None)
    (OUT/'submission_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8')
    print('OPEN_MODEL_CSV_VALID '+json.dumps(validation),flush=True)

if __name__=='__main__':main()
