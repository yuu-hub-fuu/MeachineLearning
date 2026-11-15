"""Reproduce the selected upgrade submission from checkpoints without training."""
import argparse,csv,json,gc
from pathlib import Path
from upgrade_experiments import FeatureHead,ROOT,OUT,torch,np,seed_everything
from upgrade_models import UpgradeFER,UpgradeImages


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'submission_upgrade_reproduced.csv')
    p.add_argument('--batch',type=int,default=16);args=p.parse_args();seed_everything(42)
    selection=json.loads((OUT/'model_selection.json').read_text(encoding='utf-8'))['upgrade']
    assert selection['adopted_over_core']
    data=np.load(ROOT/'data'/'dataset.npz');predictions=[]
    for record in selection['selected']:
        kind,mode=record['kind'],record['mode'];source=UpgradeFER(kind).cuda().eval();adapter=None
        if mode!='native':
            path=ROOT/'runs'/f'upgrade_{kind}_{mode}_full_s42'/'best.pt'
            ckpt=torch.load(path,map_location='cpu',weights_only=True)
            adapter=FeatureHead(source.initial_weight,source.initial_bias,mode)
            adapter.load_state_dict(ckpt['model'],strict=True);adapter=adapter.cuda().eval()
        loader=torch.utils.data.DataLoader(UpgradeImages(data['test_images'],kind),batch_size=args.batch,shuffle=False,num_workers=0)
        probabilities=[]
        for x,_ in loader:
            x=x.cuda();views=[]
            for view in range(2 if record['tta']=='flip' else 1):
                with torch.autocast('cuda',dtype=torch.bfloat16):z,f=source(x if view==0 else x.flip(-1))
                if adapter is not None:z,_=adapter(f.float().half().float())
                views.append(z.float().softmax(1))
            probabilities.append(torch.stack(views).mean(0).cpu().numpy())
        predictions.append(np.concatenate(probabilities));del source,adapter;gc.collect();torch.cuda.empty_cache()
    probabilities=np.mean(predictions,axis=0)
    assert probabilities.shape==(7065,6) and np.isfinite(probabilities).all()
    with args.output.open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['ID','Emotion']);w.writerows(zip(data['test_ids'],probabilities.argmax(1)))
    print(json.dumps({'output':str(args.output),'rows':len(probabilities)}))


if __name__=='__main__':main()
