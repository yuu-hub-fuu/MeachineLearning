"""Complete V2 inference, conservative OOF-checked component filtering, export."""
import argparse
import json
from types import SimpleNamespace
import numpy as np
import torch
import improve as imp
from postprocess_probe import refine


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cached',action='store_true',help='Re-export existing raw probabilities without GPU inference')
    args=ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    if not args.cached:
        imp.infer(SimpleNamespace(variant='control',style='d8'))
    raw=np.load(imp.V2/'probabilities.npz')
    path=imp.V2/'metrics.json'
    m=json.loads(path.read_text(encoding='utf-8'))
    _,y,_,ids=imp.arrays('train')
    threshold=m['selected']['threshold']
    original=imp.base.dice_scores(raw['oof'],y,threshold)
    candidate=refine(raw['oof'],threshold,minimum=32)
    score=imp.base.dice_scores(candidate,y,.5)
    fold_delta=[float((score-original)[v].mean()) for v in imp.base.fold_splits(ids)]
    # A single size (32) was predefined by the V1-only diagnostic. Require
    # a positive mean and broad image/fold support before using it on V2.
    use=bool(score.mean()-original.mean()>.0002 and (score>original).sum()>=12 and sum(d>0 for d in fold_delta)>=4)
    m['postprocess']=dict(minimum_component_size=32 if use else 0,connectivity=8,
                          before=float(original.mean()),candidate=float(score.mean()),
                          adopted=use,improved_images=int((score>original).sum()),fold_delta=fold_delta)
    baseline=np.load(imp.base.RUN/'probabilities.npz')['oof'].astype(np.float32)
    final=score if use else original
    m['selected']['dice']=float(final.mean())
    m['per_image']=[dict(id=name,baseline=float(a),v2=float(b),delta=float(b-a)) for name,a,b in zip(ids,imp.base.dice_scores(baseline,y,.475),final)]
    imp.base.save_json(path,m)
    if use:
        masks=refine(raw['test'],threshold,minimum=32).astype(np.float32)
        imp.export(masks,raw['test_ids'].tolist(),.5,m)
    else:
        imp.export(raw['test'],raw['test_ids'].tolist(),threshold,m)
    print('FINAL_V2 '+json.dumps(dict(selected=m['selected'],postprocess=m['postprocess'])),flush=True)


if __name__=='__main__':
    main()
