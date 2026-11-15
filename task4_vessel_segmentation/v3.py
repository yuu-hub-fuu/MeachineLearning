"""Paper-guided V3: full-resolution backbone, isolated native-core auxiliary task.

Uses the MIT-licensed official FR-UNet architecture; no upstream weights/data.
Original image-level five-fold split and submitted-mask convention are retained.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pipeline as base
from third_party.fr_unet import FR_UNet

ROOT=base.ROOT
RUN=ROOT/'v3'


def make_model(scale=2,dropout=.2):
    return FR_UNet(num_classes=2,num_channels=4,feature_scale=scale,dropout=dropout).to(device='cuda',memory_format=torch.channels_last)


@torch.inference_mode()
def predict(model,x,tta=False,core=False):
    model.eval()
    out=[]
    for im in x:
        a=torch.from_numpy(im[None]).cuda()
        parts=[]
        for k in (range(4) if tta else range(1)):
            with torch.autocast('cuda',dtype=torch.float16):
                p=model(torch.rot90(a,k,(-2,-1))).float().sigmoid()
            p=p[:,1:2] if core else p[:,:1]
            parts.append(torch.rot90(p,-k,(-2,-1)))
        out.append(torch.stack(parts).mean(0)[0,0].cpu().numpy())
    return np.stack(out)


def train(args,fold,arm):
    x,y,native,ids=base.load_arrays('train')
    targets=np.concatenate([y,native],1)
    valid=base.fold_splits(ids)[fold]
    tr=[i for i in range(20) if i not in valid]
    tx,ty=x[tr],targets[tr]
    seed=20261103+fold*100
    base.seed_all(seed)
    rng=np.random.default_rng(seed)
    model=make_model(args.scale,args.dropout)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.0001)
    scaler=torch.amp.GradScaler('cuda')
    out=RUN/arm/f'fold{fold}'
    out.mkdir(parents=True,exist_ok=True)
    best=-1.;best_step=0;start_step=0;elapsed=0.;history=[]
    latest=out/'latest.pt'
    if args.resume and latest.exists():
        ck=torch.load(latest,map_location='cuda',weights_only=False)
        assert ck['args']['steps']==args.steps and ck['args']['scale']==args.scale
        assert ck['args']['patch']==args.patch and ck['args']['batch']==args.batch
        assert ck['arm']==arm and ck['fold']==fold
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler'])
        rng.bit_generator.state=ck['rng_numpy']
        torch.set_rng_state(ck['rng_cpu'].cpu())
        torch.cuda.set_rng_state_all([v.cpu() for v in ck['rng_cuda']])
        start_step=ck['step'];best=ck['best'];best_step=ck['best_step'];elapsed=ck['elapsed'];history=ck['history']
        print(f'RESUME {arm} fold={fold} step={start_step}',flush=True)
    elif latest.exists():
        raise RuntimeError(f'{latest} already exists; use --resume or a new arm directory')
    base.save_json(out/'config.json',dict(**vars(args),fold=fold,arm=arm,seed=seed,
                   train_ids=[ids[i] for i in tr],valid_ids=[ids[i] for i in valid],
                   main_target='PIL bilinear 512 then >0',auxiliary_target='PIL nearest 512 native vessel labels',
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   architecture_sha256=hashlib.sha256((ROOT/'third_party/fr_unet/fr_unet.py').read_bytes()).hexdigest(),
                   parameters=sum(p.numel() for p in model.parameters()),
                   optimizer='AdamW',lr_end=.00002,core_weight=.3 if arm=='core' else 0.,
                   no_external_retinal_data_or_weights=True))
    stop=min(args.stop_at or args.steps,args.steps)
    start=time.time()
    for step in range(start_step+1,stop+1):
        model.train()
        xx,yy=base.sample_batch(tx,ty,args.batch,args.patch,rng)
        xx=xx.contiguous(memory_format=torch.channels_last)
        warmup=min(step/50,1.)
        lr=(.00002+(args.lr-.00002)*.5*(1+math.cos(math.pi*(step-1)/args.steps)))*warmup
        for pg in opt.param_groups:pg['lr']=lr
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            logits=model(xx)
            main_loss=base.loss_fn(logits[:,:1],yy[:,:1])
            aux_loss=base.loss_fn(logits[:,1:2],yy[:,1:2]) if arm=='core' else logits[:,1:2].sum()*0
            loss=main_loss+(.3*aux_loss if arm=='core' else 0)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite V3 training loss')
        scaler.scale(loss).backward();scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(),5)
        scaler.step(opt);scaler.update()
        if step%args.eval_every==0 or step==stop:
            pp=predict(model,x[valid])
            score=float(base.dice_scores(pp,y[valid],.5).mean())
            if score>best:
                best=score;best_step=step
                torch.save(dict(model=model.state_dict(),scale=args.scale,dropout=args.dropout,
                                step=step,fold=fold,arm=arm,seed=seed,dice=score),out/'best.pt')
            row=dict(arm=arm,fold=fold,step=step,total=args.steps,stop_at=stop,dice=score,best=best,best_step=best_step,
                     loss=float(loss.detach()),main_loss=float(main_loss.detach()),aux_loss=float(aux_loss.detach()),
                     lr=lr,seconds=round(elapsed+time.time()-start,2),gpu_peak_mb=round(torch.cuda.max_memory_allocated()/2**20))
            history.append(row)
            base.save_json(out/'history.json',history);base.save_json(RUN/'status.json',row)
            print(json.dumps(row),flush=True)
            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),
                            step=step,best=best,best_step=best_step,fold=fold,arm=arm,args=vars(args),
                            rng_numpy=rng.bit_generator.state,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),
                            elapsed=row['seconds'],history=history),latest)
    ck=torch.load(out/'best.pt',map_location='cuda',weights_only=False)
    model.load_state_dict(ck['model'])
    pp=predict(model,x[valid],tta=True)
    core=predict(model,x[valid],tta=True,core=True) if arm=='core' else np.zeros_like(pp)
    np.savez_compressed(out/'validation.npz',ids=np.array([ids[i] for i in valid]),probabilities=pp.astype(np.float16),core=core.astype(np.float16))
    summary=dict(arm=arm,fold=fold,trained_step=stop,best_step=best_step,dice=float(base.dice_scores(pp,y[valid],.5).mean()),
                 best_no_tta=best,core_dice=float(base.dice_scores(core,native[valid],.5).mean()) if arm=='core' else None,
                 valid_ids=[ids[i] for i in valid],seconds=round(elapsed+time.time()-start,2))
    base.save_json(out/'summary.json',summary)
    print('V3_FOLD '+json.dumps(summary),flush=True)
    del model,opt,scaler
    torch.cuda.empty_cache()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('mode',choices=['train'])
    ap.add_argument('--folds',nargs='+',type=int,default=[2])
    ap.add_argument('--arms',nargs='+',choices=['plain','core'],default=['plain','core'])
    ap.add_argument('--steps',type=int,default=2600)
    ap.add_argument('--stop-at',type=int,default=1200)
    ap.add_argument('--eval-every',type=int,default=200)
    ap.add_argument('--batch',type=int,default=8)
    ap.add_argument('--patch',type=int,default=128)
    ap.add_argument('--scale',type=int,default=2)
    ap.add_argument('--dropout',type=float,default=.2)
    ap.add_argument('--lr',type=float,default=.0007)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=True
    torch.use_deterministic_algorithms(False)
    RUN.mkdir(exist_ok=True)
    for fold in args.folds:
        for arm in args.arms:train(args,fold,arm)


if __name__=='__main__':main()
