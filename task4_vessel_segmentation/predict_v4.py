"""Complete five-fold transfer/native-grid inference, OOF calibration and CSV."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,csv,hashlib,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
import pipeline as base
import v3_native as native
from v4_transfer import TransferUNet
from operator_confidence import lift

ROOT=base.ROOT
RUN=ROOT/'v4'


def export(p,ids,threshold,metrics):
    dest=RUN/'delivery';white=dest/'vessel_white';black=dest/'vessel_black'
    white.mkdir(parents=True,exist_ok=True);black.mkdir(parents=True,exist_ok=True)
    for name,a in zip(ids,p):
        m=(a>=threshold).astype(np.uint8)*255
        Image.fromarray(m).save(white/f'{name}.png');Image.fromarray(255-m).save(black/f'{name}.png')
    stage=RUN/'official_export';(stage/'image').mkdir(parents=True,exist_ok=True)
    for file in white.glob('*.png'):shutil.copy2(file,stage/'image'/file.name)
    shutil.copy2(base.DATA/'segmentation_to_csv.py',stage/'segmentation_to_csv.py')
    subprocess.run([sys.executable,str(stage/'segmentation_to_csv.py')],cwd=stage,check=True)
    with (stage/'submission.csv').open(encoding='utf-8',newline='') as f:rows={r['Id']:r['Predicted'] for r in csv.DictReader(f)}
    with (base.DATA/'sample_submission.csv').open(encoding='utf-8',newline='') as f:order=[r['Id'] for r in csv.DictReader(f)]
    assert len(rows)==20 and set(rows)==set(order)==set(ids)
    stats=[]
    with (dest/'submission_v4.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['Id','Predicted'])
        for name in order:
            mask=np.asarray(Image.open(white/f'{name}.png'))>0
            assert mask.shape==(512,512)
            assert rows[name]==base.encode(mask)
            assert np.array_equal(base.decode(rows[name]),mask)
            assert np.array_equal(np.asarray(Image.open(black/f'{name}.png')),(~mask).astype(np.uint8)*255)
            assert 0<mask.mean()<.5
            w.writerow([name,rows[name]])
            stats.append(dict(id=name,vessel_fraction=float(mask.mean())))
    base.save_json(dest/'submission_validation.json',dict(valid=True,rows=20,
                   sha256=hashlib.sha256((dest/'submission_v4.csv').read_bytes()).hexdigest(),images=stats,
                   target='white vessel foreground; official-script generated and independently decoded',selected=metrics['selected']))
    print('V4_CSV_VALID',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cached',action='store_true');args=ap.parse_args()
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    _,target,_,ids=base.load_arrays('train')
    old=np.load(ROOT/'v2/probabilities.npz')
    assert old['train_ids'].tolist()==ids
    pold=old['oof'].astype(np.float32);told=old['test'].astype(np.float32)
    cache=RUN/'lifted_confidence.npz'
    if args.cached:
        z=np.load(cache);oof=z['oof'];test=z['test'];test_ids=z['test_ids'].tolist()
    else:
        x,_,ids2=native.load_native('train');xt,_,test_ids=native.load_native('test');assert ids==ids2
        oof_native=np.zeros((20,584,565),np.float32);test_native=np.zeros_like(oof_native)
        test_lift=np.zeros((20,512,512),np.float32);seen=set()
        for fold,valid in enumerate(base.fold_splits(ids)):
            config=json.loads((RUN/'transfer'/f'fold{fold}'/'config.json').read_text(encoding='utf-8'))
            assert not set(config['train_ids'])&set(config['valid_ids'])
            assert config['valid_ids']==[ids[i] for i in valid]
            ck=torch.load(RUN/'transfer'/f'fold{fold}'/'best.pt',map_location='cuda',weights_only=False)
            model=TransferUNet(pretrained=False).to(device='cuda',memory_format=torch.channels_last)
            model.load_state_dict(ck['model'])
            assert not seen.intersection(valid);seen.update(valid)
            oof_native[valid]=native.predict(model,x[valid],tta=True)
            individual_test=native.predict(model,xt,tta=True)
            test_native+=individual_test/5
            # The conversion is nonlinear. Each member must enter the same
            # submitted-mask space BEFORE model averaging, matching OOF f_k(x).
            test_lift+=lift(individual_test)/5
            print('V4_INFERENCE_FOLD '+str(fold),flush=True)
            del model;torch.cuda.empty_cache()
        assert len(seen)==20 and np.isfinite(oof_native).all() and np.isfinite(test_native).all()
        np.savez_compressed(RUN/'native_probabilities.npz',train_ids=np.array(ids),oof=oof_native,test_ids=np.array(test_ids),test=test_native)
        oof=lift(oof_native);test=test_lift
        np.savez_compressed(cache,train_ids=np.array(ids),oof=oof,test_ids=np.array(test_ids),test=test)
    assert test_ids==old['test_ids'].tolist()
    assert oof.shape==test.shape==(20,512,512) and np.isfinite(oof).all() and np.isfinite(test).all()
    candidates=[]
    for alpha in (0.,.25,.5,.75,1.):
        p=alpha*oof+(1-alpha)*pold
        for t in np.round(np.arange(.2,.701,.025),3):
            score=float(base.dice_scores(p,target,t).mean())
            candidates.append(dict(alpha=alpha,threshold=float(t),dice=score))
    best=max(candidates,key=lambda r:r['dice'])
    p=best['alpha']*oof+(1-best['alpha'])*pold
    pt=best['alpha']*test+(1-best['alpha'])*told
    before=base.dice_scores(pold,target,.5);after=base.dice_scores(p,target,best['threshold'])
    family=max((r for r in candidates if r['alpha']==1.),key=lambda r:r['dice'])
    metrics=dict(baseline_v1_online_user_reported=.85042,baseline_v2_oof=float(before.mean()),
                 new_family_oof=family,selected=best,delta_vs_v2=float(after.mean()-before.mean()),
                 images_improved=int((after>before).sum()),candidates=candidates,
                 per_image=[dict(id=name,baseline_v2=float(a),v4=float(b),delta=float(b-a)) for name,a,b in zip(ids,before,after)],
                 fold_comparison=[dict(fold=f,baseline_v2=float(before[v].mean()),v4=float(after[v].mean()),delta=float((after-before)[v].mean())) for f,v in enumerate(base.fold_splits(ids))],
                 protocol='native vessel learning; exact binary conversion averaged over 100 fixed levels; global OOF mixture and threshold selection',
                 note='All values are model-selection validation estimates. No new Kaggle score is available. Test labels were never used.',
                 inference_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    base.save_json(RUN/'metrics.json',metrics)
    np.savez_compressed(RUN/'selected_probabilities.npz',train_ids=np.array(ids),oof=p,test_ids=np.array(test_ids),test=pt)
    export(pt,test_ids,best['threshold'],metrics)
    # Include the worst regression and the largest gain in validation previews.
    dest=RUN/'delivery';sheet=Image.new('RGB',(1024,1024),'black');draw=ImageDraw.Draw(sheet)
    order=np.argsort(after-before);selected=[int(order[0]),int(order[1]),int(order[-2]),int(order[-1])]
    for row,i in enumerate(selected):
        orig=Image.open(base.DATA/'train/image'/f'{ids[i]}.jpg').convert('RGB')
        gt=Image.fromarray((target[i,0]*255).astype(np.uint8))
        oldm=Image.fromarray((pold[i]>=.5).astype(np.uint8)*255)
        newm=Image.fromarray((p[i]>=best['threshold']).astype(np.uint8)*255)
        for col,im in enumerate([orig,gt,oldm,newm]):sheet.paste(im.convert('RGB').resize((256,256)),(col*256,row*256))
        draw.text((3,row*256+3),f'OOF {ids[i]} delta={after[i]-before[i]:+.4f}',fill='white')
    sheet.save(dest/'validation_comparison.png')
    preview=Image.new('RGB',(768,1024),'black')
    for row,i in enumerate([0,4,9,19]):
        name=test_ids[i]
        images=[Image.open(base.DATA/'test/image'/f'{name}.jpg'),Image.open(ROOT/'v2/delivery/vessel_white'/f'{name}.png'),Image.open(dest/'vessel_white'/f'{name}.png')]
        for col,im in enumerate(images):preview.paste(im.convert('RGB').resize((256,256)),(col*256,row*256))
    preview.save(dest/'test_comparison.png')
    base.save_json(RUN/'status.json',dict(state='complete',**best,submission=str(dest/'submission_v4.csv')))
    print('V4_COMPLETE '+json.dumps(dict(selected=best,delta=metrics['delta_vs_v2'],improved=metrics['images_improved'])),flush=True)


if __name__=='__main__':main()
