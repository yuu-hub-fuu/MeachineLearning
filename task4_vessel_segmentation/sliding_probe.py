"""Validate full-image vs training-sized overlapping inference, not an assumption."""
import argparse,json,math
import numpy as np
import torch
from torch.nn import functional as F
import pipeline as base
import v3
import v3_native as native
from v4_transfer import TransferUNet


@torch.inference_mode()
def tiled(model,x,patch,stride=None,batch=8):
    stride=stride or patch//2
    model.eval();result=[]
    win=torch.hann_window(patch,periodic=False,device='cuda').clamp_min(.01)
    weight=(win[:,None]*win[None,:])[None,None]
    for im in x:
        a=torch.from_numpy(im[None]).cuda()
        h,w=a.shape[-2:]
        hh=max(patch,math.ceil((h-patch)/stride)*stride+patch)
        ww=max(patch,math.ceil((w-patch)/stride)*stride+patch)
        a=F.pad(a,(0,ww-w,0,hh-h),mode='replicate')
        coords=[(top,left) for top in range(0,hh-patch+1,stride) for left in range(0,ww-patch+1,stride)]
        accum=torch.zeros((1,1,hh,ww),device='cuda');counts=torch.zeros_like(accum)
        for start in range(0,len(coords),batch):
            positions=coords[start:start+batch]
            patches=torch.cat([a[...,t:t+patch,l:l+patch] for t,l in positions],0).contiguous(memory_format=torch.channels_last)
            with torch.autocast('cuda',dtype=torch.float16):
                ps=model(patches)[:,:1].float().sigmoid()
            for k,(t,l) in enumerate(positions):
                accum[...,t:t+patch,l:l+patch]+=ps[k:k+1]*weight
                counts[...,t:t+patch,l:l+patch]+=weight
        p=(accum/counts.clamp_min(1e-6))[0,0,:h,:w].cpu().numpy()
        result.append(p)
    return np.stack(result)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--family',choices=['transfer','frnative','plain','core'],default='transfer')
    ap.add_argument('--fold',type=int,default=2);args=ap.parse_args()
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    x,y,_,ids=base.load_arrays('train');valid=base.fold_splits(ids)[args.fold]
    if args.family=='transfer':
        path=base.ROOT/'v4/transfer'/f'fold{args.fold}'/'best.pt'
        model=TransferUNet(pretrained=False).to(device='cuda',memory_format=torch.channels_last);patch=160
    else:
        path=base.ROOT/'v3'/('native' if args.family=='frnative' else args.family)/f'fold{args.fold}'/'best.pt'
        model=v3.make_model();patch=128
    ck=torch.load(path,map_location='cuda',weights_only=False);model.load_state_dict(ck['model'])
    is_native=args.family in ('transfer','frnative')
    if is_native:x,_,ids2=native.load_native('train');assert ids==ids2
    p=tiled(model,x[valid],patch)
    full=native.predict(model,x[valid]) if is_native else v3.predict(model,x[valid])
    if is_native:
        mask=native.convert_masks(p,.5);fullmask=native.convert_masks(full,.5)
    else:mask=p>=.5;fullmask=full>=.5
    summary=dict(family=args.family,fold=args.fold,patch=patch,checkpoint_step=ck['step'],
                 whole_dice=float(base.dice_scores(fullmask,y[valid],.5).mean()),tiled_dice=float(base.dice_scores(mask,y[valid],.5).mean()))
    out=base.ROOT/'research-evaluations'/f'sliding_{args.family}_fold{args.fold}.json'
    base.save_json(out,summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
