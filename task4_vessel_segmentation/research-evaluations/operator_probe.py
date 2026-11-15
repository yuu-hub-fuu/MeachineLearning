"""Check the deterministic mask conversion separately from segmentation."""
from pathlib import Path
import sys,json
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pipeline as base


def convert512core(mask):
    # Only a diagnostic approximation to a native prediction: invert the nearest
    # resize to native shape, then apply the supplied bilinear/nonzero operator.
    a=Image.fromarray(mask.astype(np.uint8)*255).resize((565,584),Image.Resampling.NEAREST)
    return np.array(a.resize((512,512),Image.Resampling.BILINEAR))>0


def main():
    _,y,core,ids=base.load_arrays('train')
    p=np.load(base.ROOT/'v2/probabilities.npz')['oof'].astype(np.float32)
    result={}
    reconstructed=np.stack([convert512core(a[0]>.5) for a in core])
    result['label_assisted_roundtrip_upper_bound']=float(base.dice_scores(reconstructed,y,.5).mean())
    trials=[]
    for t in (.6,.7,.75,.8,.85,.9,.925,.95):
        pred=np.stack([convert512core(a>=t) for a in p])
        scores=base.dice_scores(pred,y,.5)
        trials.append(dict(core_threshold=t,dice=float(scores.mean()),per_image=scores.tolist()))
    result['v2_predictions_reconverted']=trials
    result['note']='The first value uses validation labels and is a diagnostic only. No test labels. Re-conversion of 512 masks is an approximation, not native-resolution inference.'
    base.save_json(Path(__file__).parent/'operator_probe.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
