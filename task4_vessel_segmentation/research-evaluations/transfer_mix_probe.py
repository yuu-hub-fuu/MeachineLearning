from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pipeline as b
from operator_confidence import lift


def main():
    _,y,_,ids=b.load_arrays('train')
    old=np.load(b.ROOT/'v2/probabilities.npz')['oof'].astype(np.float32)
    pp=[];ix=[]
    for fold in [0,2]:
        z=np.load(b.ROOT/'v4/transfer'/f'fold{fold}'/'validation.npz')
        pp.extend(z['probabilities'].astype(np.float32));ix.extend(ids.index(str(n)) for n in z['ids'])
    new=lift(np.stack(pp))
    rows=[]
    for alpha in [0,.25,.5,.75,1.]:
        mix=alpha*new+(1-alpha)*old[ix]
        for t in np.round(np.arange(.2,.701,.025),3):
            scores=b.dice_scores(mix,y[ix],t)
            rows.append(dict(alpha=alpha,threshold=float(t),dice=float(scores.mean()),fold0=float(scores[:4].mean()),fold2=float(scores[4:].mean())))
    rows.sort(key=lambda r:r['dice'],reverse=True)
    b.save_json(b.ROOT/'research-evaluations/transfer_mix_probe.json',dict(best=rows[0],top=rows[:20],baseline=float(b.dice_scores(old[ix],y[ix],.5).mean())))
    np.savez_compressed(b.ROOT/'research-evaluations/transfer_twofold_lift.npz',indices=np.array(ix),confidence=new)
    print(json.dumps(rows[:5],indent=2))


if __name__=='__main__':main()
