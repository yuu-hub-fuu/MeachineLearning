"""OOF-only, small predefined connectivity and component-size diagnostics."""
import json
import numpy as np
from scipy import ndimage as ndi
import pipeline as base


def refine(p, threshold, minimum=0, high=None):
    result=[]
    structure=np.ones((3,3),bool)
    for pp in p:
        mask=pp>=threshold
        if high is not None:
            mask=ndi.binary_propagation(pp>=high,structure=structure,mask=mask)
        if minimum:
            labels,n=ndi.label(mask,structure=structure)
            sizes=np.bincount(labels.reshape(-1))
            keep=sizes>=minimum
            keep[0]=False
            mask=keep[labels]
        result.append(mask)
    return np.stack(result)


def main():
    p=np.load(base.RUN/'probabilities.npz')['oof'].astype(np.float32)
    _,y,_,ids=base.load_arrays('train')
    baseline=base.dice_scores(p,y,.475)
    rows=[]
    for t in (.4,.425,.45,.475,.5,.525,.55):
        for minimum in (0,8,16,32,64):
            mask=refine(p,t,minimum)
            score=base.dice_scores(mask,y,.5)
            rows.append(dict(kind='components',threshold=t,minimum=minimum,dice=float(score.mean()),
                             improved=int((score>baseline).sum()),fold_delta=[float((score-baseline)[v].mean()) for v in base.fold_splits(ids)]))
    for low,high in ((.35,.55),(.4,.55),(.425,.55),(.45,.55),(.45,.6),(.475,.6)):
        mask=refine(p,low,high=high)
        score=base.dice_scores(mask,y,.5)
        rows.append(dict(kind='hysteresis',threshold=low,high=high,dice=float(score.mean()),
                         improved=int((score>baseline).sum()),fold_delta=[float((score-baseline)[v].mean()) for v in base.fold_splits(ids)]))
    rows.sort(key=lambda r:r['dice'],reverse=True)
    base.save_json(base.ROOT/'v2/postprocess_probe.json',dict(baseline=float(baseline.mean()),trials=rows))
    print(json.dumps(rows[:8],indent=2))


if __name__=='__main__':
    main()
