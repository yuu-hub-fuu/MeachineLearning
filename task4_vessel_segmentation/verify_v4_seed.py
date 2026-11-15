"""Independent training seed diagnostic, using the already fixed global rule."""
import json,numpy as np,torch
import pipeline as base
import v3_native as native
from v4_transfer import TransferUNet
from operator_confidence import lift


def main():
    root=base.ROOT;out=root/'v4'
    m=json.loads((out/'metrics.json').read_text(encoding='utf-8'))
    x,_,ids=native.load_native('train');_,y,_,_=base.load_arrays('train')
    valid=base.fold_splits(ids)[2]
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    ck=torch.load(out/'transfer/seed10000/fold2/best.pt',map_location='cuda',weights_only=False)
    model=TransferUNet(pretrained=False).to(device='cuda',memory_format=torch.channels_last)
    model.load_state_dict(ck['model'])
    pp=lift(native.predict(model,x[valid],tta=True))
    old=np.load(root/'v2/probabilities.npz')['oof'][valid].astype(np.float32)
    a=m['selected']['alpha'];t=m['selected']['threshold']
    score=float(base.dice_scores(a*pp+(1-a)*old,y[valid],t).mean())
    initial=next(r for r in m['fold_comparison'] if r['fold']==2)
    result=dict(fold=2,alternate_seed_offset=10000,selected_rule=dict(alpha=a,threshold=t),
                baseline_v2=initial['baseline_v2'],main_seed=initial['v4'],alternate_seed=score,
                alternate_delta_vs_v2=score-initial['baseline_v2'],
                note='Same global rule, no threshold retuning for this seed. Diagnostic only; alternate weights are not used in test submission.')
    base.save_json(out/'seed_check.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
