"""Native-grid vessel learning followed by the exact submitted-mask operator.

Training labels are the unresized supplied binary masks. Validation and export
apply the supplied PIL-bilinear-to-512 + nonzero operator AFTER binarization.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,math,time,inspect
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import torch
from torch.nn import functional as F
import pipeline as base
import v3

RUN=base.ROOT/'v3/native'


def load_native(split):
    images=[];labels=[];ids=[]
    for p in sorted((base.DATA/split/'image').glob('*.jpg'),key=lambda p:int(p.stem)):
        im=np.array(Image.open(p).convert('RGB'),dtype=np.float32)/255
        g=im[...,1]
        local=np.clip((g-gaussian_filter(g,8))*4+.5,0,1)
        images.append(np.concatenate([im,local[...,None]],-1).transpose(2,0,1))
        ids.append(p.stem)
        if split=='train':labels.append((np.array(Image.open(base.DATA/split/'label'/p.name).convert('L'))>0).astype(np.float32)[None])
    return np.ascontiguousarray(np.stack(images)),np.stack(labels) if labels else None,ids


def convert_masks(p,threshold=.5):
    return np.stack([np.asarray(Image.fromarray((a>=threshold).astype(np.uint8)*255).resize((512,512),Image.Resampling.BILINEAR))>0 for a in p])


@torch.inference_mode()
def predict(model,x,tta=False):
    model.eval();result=[]
    for im in x:
        a=torch.from_numpy(im[None]).cuda()
        h,w=a.shape[-2:]
        a=F.pad(a,(0,(-w)%8,0,(-h)%8),mode='replicate')
        parts=[]
        for k in (range(4) if tta else range(1)):
            with torch.autocast('cuda',dtype=torch.float16):
                p=model(torch.rot90(a,k,(-2,-1)))[:,:1].float().sigmoid()
            parts.append(torch.rot90(p,-k,(-2,-1))[...,:h,:w])
        result.append(torch.stack(parts).mean(0)[0,0].cpu().numpy())
    return np.stack(result)


def sample(x,y,rng,batch,patch):
    xx=[];yy=[]
    for i in rng.integers(len(x),size=batch):
        top=int(rng.integers(x.shape[-2]-patch+1));left=int(rng.integers(x.shape[-1]-patch+1))
        a=x[i,:,top:top+patch,left:left+patch];b=y[i,:,top:top+patch,left:left+patch]
        k=int(rng.integers(4));a=np.rot90(a,k,(-2,-1));b=np.rot90(b,k,(-2,-1))
        if rng.random()<.5:a=a[...,::-1];b=b[...,::-1]
        xx.append(a);yy.append(b)
    a=torch.from_numpy(np.ascontiguousarray(np.stack(xx))).cuda().contiguous(memory_format=torch.channels_last)
    b=torch.from_numpy(np.ascontiguousarray(np.stack(yy))).cuda()
    scale=torch.empty((batch,4,1,1),device='cuda').uniform_(.85,1.15)
    shift=torch.empty((batch,4,1,1),device='cuda').uniform_(-.05,.05)
    a=(a*scale+shift).clamp(0,1)
    if rng.random()<.2:a=(a+torch.randn_like(a)*.015).clamp(0,1)
    return a,b


def train(args,fold,model_factory=None):
    x,y,ids=load_native('train')
    _,target,_,id_check=base.load_arrays('train')
    assert ids==id_check
    valid=base.fold_splits(ids)[fold];tr=[i for i in range(20) if i not in valid]
    seed=20261103+fold*100+getattr(args,'seed_offset',0)
    base.seed_all(seed);rng=np.random.default_rng(seed)
    model=v3.make_model(args.scale,.2) if model_factory is None else model_factory(args)
    if hasattr(model,'encoder'):
        encoder_ids={id(p) for p in model.encoder.parameters()}
        groups=[dict(params=list(model.encoder.parameters()),lr_scale=.1),
                dict(params=[p for p in model.parameters() if id(p) not in encoder_ids],lr_scale=1.)]
    else:groups=[dict(params=list(model.parameters()),lr_scale=1.)]
    opt=torch.optim.AdamW(groups,lr=args.lr,weight_decay=.0001)
    scaler=torch.amp.GradScaler('cuda')
    out=RUN/f'fold{fold}';out.mkdir(parents=True,exist_ok=True)
    latest=out/'latest.pt'
    best=-1.;best_step=0;start_step=0;elapsed=0.;history=[]
    if args.resume and latest.exists():
        ck=torch.load(latest,map_location='cuda',weights_only=False)
        assert ck['args']['steps']==args.steps and ck['args']['scale']==args.scale
        assert ck['args']['batch']==args.batch and ck['args']['patch']==args.patch
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler'])
        rng.bit_generator.state=ck['rng_numpy'];torch.set_rng_state(ck['rng_cpu'].cpu())
        torch.cuda.set_rng_state_all([v.cpu() for v in ck['rng_cuda']])
        start_step=ck['step'];best=ck['best'];best_step=ck['best_step'];elapsed=ck['elapsed'];history=ck['history']
    elif latest.exists():raise RuntimeError('Native run exists; use --resume')
    base.save_json(out/'config.json',dict(**vars(args),fold=fold,seed=seed,
                   train_ids=[ids[i] for i in tr],valid_ids=[ids[i] for i in valid],
                   training_target='supplied native binary 584x565 masks; no label resize',
                   validation_operator='native sigmoid >=0.5, binary PNG, PIL bilinear512, >0',
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   model_class=type(model).__name__,
                   architecture_sha256=hashlib.sha256(Path(inspect.getfile(type(model))).read_bytes()).hexdigest(),
                   no_external_retinal_data_or_weights=True))
    stop=min(args.stop_at or args.steps,args.steps)
    tx,ty=x[tr],y[tr]
    start=time.time()
    for step in range(start_step+1,stop+1):
        model.train();a,b=sample(tx,ty,rng,args.batch,args.patch)
        lr=(.00002+(args.lr-.00002)*.5*(1+math.cos(math.pi*(step-1)/args.steps)))*min(step/50,1.)
        for pg in opt.param_groups:pg['lr']=lr*pg.get('lr_scale',1.)
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):loss=base.loss_fn(model(a)[:,:1],b)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite native loss')
        scaler.scale(loss).backward();scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(),5)
        scaler.step(opt);scaler.update()
        if step%200==0 or step==stop:
            p=predict(model,x[valid]);mask=convert_masks(p,.5)
            score=float(base.dice_scores(mask,target[valid],.5).mean())
            native_score=float(base.dice_scores(p,y[valid],.5).mean())
            if score>best:
                best=score;best_step=step
                torch.save(dict(model=model.state_dict(),scale=args.scale,dropout=.2,fold=fold,step=step,dice=score,model_class=type(model).__name__),out/'best.pt')
            row=dict(arm='native',fold=fold,step=step,total=args.steps,stop_at=stop,dice=score,native_dice=native_score,
                     best=best,best_step=best_step,loss=float(loss.detach()),lr=lr,seconds=round(elapsed+time.time()-start,2))
            history.append(row);base.save_json(out/'history.json',history);base.save_json(base.ROOT/'v3/status.json',row)
            print(json.dumps(row),flush=True)
            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),
                            args=vars(args),step=step,best=best,best_step=best_step,elapsed=row['seconds'],history=history,
                            rng_numpy=rng.bit_generator.state,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all()),latest)
    ck=torch.load(out/'best.pt',map_location='cuda',weights_only=False);model.load_state_dict(ck['model'])
    p=predict(model,x[valid],tta=True)
    np.savez_compressed(out/'validation.npz',ids=np.array([ids[i] for i in valid]),probabilities=p.astype(np.float16))
    summary=dict(arm='native',fold=fold,best_step=best_step,trained_step=stop,
                 dice=float(base.dice_scores(convert_masks(p,.5),target[valid],.5).mean()),
                 native_dice=float(base.dice_scores(p,y[valid],.5).mean()),valid_ids=[ids[i] for i in valid])
    base.save_json(out/'summary.json',summary);print('NATIVE_FOLD '+json.dumps(summary),flush=True)
    del model,opt,scaler;torch.cuda.empty_cache()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--folds',type=int,nargs='+',default=[2])
    ap.add_argument('--steps',type=int,default=2600);ap.add_argument('--stop-at',type=int,default=1200)
    ap.add_argument('--batch',type=int,default=8);ap.add_argument('--patch',type=int,default=128)
    ap.add_argument('--scale',type=int,default=2);ap.add_argument('--lr',type=float,default=.0007)
    ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True;torch.use_deterministic_algorithms(False)
    for fold in args.folds:train(args,fold)


if __name__=='__main__':main()
