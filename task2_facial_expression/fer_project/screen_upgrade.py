"""Evaluate available author weights immediately on the fixed validation split."""
import argparse,json,time
from upgrade_experiments import ROOT,OUT,score,dump,seed_everything,torch,np
from upgrade_models import UpgradeFER,UpgradeImages

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',required=True);p.add_argument('--batch',type=int,default=16);a=p.parse_args()
    seed_everything(42);data=np.load(ROOT/'data'/'dataset.npz');va=data['val_idx'];y=data['labels'][va]
    model=UpgradeFER(a.kind).cuda().eval();pred=[];started=time.time()
    dl=torch.utils.data.DataLoader(UpgradeImages(data['train_images'][va],a.kind),batch_size=a.batch,num_workers=0)
    for step,(x,index) in enumerate(dl):
        x=x.cuda();ps=[]
        for flip in [False,True]:
            with torch.autocast('cuda',dtype=torch.bfloat16):z,_=model(x.flip(-1) if flip else x)
            ps.append(z.float().softmax(1).cpu().numpy())
        pred.append(np.stack(ps,axis=1))
        if step%50==0:print(json.dumps({'event':'screen_progress','kind':a.kind,'images':int(index[-1])+1,'seconds':time.time()-started}),flush=True)
    pred=np.concatenate(pred);run=ROOT/'runs'/f'upgrade_{a.kind}_screen';run.mkdir(exist_ok=True)
    np.savez(run/'validation.npz',indices=va,labels=y,probabilities=pred)
    report={'kind':a.kind,'single':score(pred[:,0],y),'flip':score(pred.mean(1),y),'seconds':time.time()-started,
            'note':'Frozen author weights before task adaptation; potential source overlap for QCS/RMN; not Kaggle score'}
    dump(OUT/f'upgrade_{a.kind}_screen.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
