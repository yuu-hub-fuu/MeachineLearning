"""Read-only OOF error audit for choosing architecture/loss changes."""
from pathlib import Path
import json
import sys
import numpy as np
from scipy import ndimage as ndi
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pipeline as base


def skeleton(mask):
    """Zhang-Suen thinning, used for diagnostics only, never as predictions."""
    a=np.pad(mask.astype(bool),1)
    for _ in range(256):
        changed=False
        for phase in (0,1):
            q=[a[:-2,1:-1],a[:-2,2:],a[1:-1,2:],a[2:,2:],a[2:,1:-1],a[2:,:-2],a[1:-1,:-2],a[:-2,:-2]]
            center=a[1:-1,1:-1]
            n=sum(x.astype(np.uint8) for x in q)
            transitions=sum((~q[k]&q[(k+1)%8]).astype(np.uint8) for k in range(8))
            if phase==0:
                c1=~(q[0]&q[2]&q[4]);c2=~(q[2]&q[4]&q[6])
            else:
                c1=~(q[0]&q[2]&q[6]);c2=~(q[0]&q[4]&q[6])
            remove=center&(n>=2)&(n<=6)&(transitions==1)&c1&c2
            if remove.any():
                center[remove]=False
                changed=True
        if not changed:
            break
    return a[1:-1,1:-1]


def div(a,b):
    return float(a/max(b,1))


def main():
    x,y,native,ids=base.load_arrays('train')
    p=np.load(base.ROOT/'v2/probabilities.npz')['oof'].astype(np.float32)
    pred=p>=.5
    gt=y[:,0]>.5
    orig=native[:,0]>.5
    annulus=gt&~orig
    fn=gt&~pred
    fp=pred&~gt
    rows=[]
    sums={k:0 for k in ['thin_n','thin_hit','mid_n','mid_hit','thick_n','thick_hit','skel_n','skel_hit','fp_near','fp_far']}
    for i,name in enumerate(ids):
        sk=skeleton(orig[i])
        radius=ndi.distance_transform_edt(orig[i])
        gt_dist=ndi.distance_transform_edt(~gt[i])
        row=dict(id=name,dice=float(base.dice_scores(p[i:i+1],y[i:i+1],.5)[0]),
                 native_vessel_pixels=int(orig[i].sum()),official_vessel_pixels=int(gt[i].sum()),
                 added_boundary_pixels=int(annulus[i].sum()),
                 false_negative_core=int((fn[i]&orig[i]).sum()),false_negative_added_boundary=int((fn[i]&annulus[i]).sum()),
                 false_positive_near_boundary=int((fp[i]&(gt_dist<=2)).sum()),
                 false_positive_far=int((fp[i]&(gt_dist>2)).sum()))
        for label,lo,hi in [('thin',0,1.5),('mid',1.5,3),('thick',3,100)]:
            subset=sk&(radius>lo)&(radius<=hi)
            n=int(subset.sum());hit=int((subset&pred[i]).sum())
            row[label+'_centerline_pixels']=n
            row[label+'_centerline_recall']=div(hit,n)
            sums[label+'_n']+=n;sums[label+'_hit']+=hit
        row['centerline_recall']=div((sk&pred[i]).sum(),sk.sum())
        row['centerline_recall_tolerance1']=div((sk&ndi.binary_dilation(pred[i])).sum(),sk.sum())
        sums['skel_n']+=int(sk.sum());sums['skel_hit']+=int((sk&pred[i]).sum())
        sums['fp_near']+=row['false_positive_near_boundary'];sums['fp_far']+=row['false_positive_far']
        rows.append(row)
    b=float(base.dice_scores(p,y,.5).mean())
    # These label-assisted counterfactuals are upper-bound diagnostics, not valid
    # predictions or candidate submissions. No test labels are read.
    corrected_boundary=pred|(fn&annulus)
    corrected_core=pred|(fn&orig)
    s=dict(oof_dice=b,images=20,
           native_vessel_fraction=float(orig.mean()),official_vessel_fraction=float(gt.mean()),
           official_vessel_area_increase_fraction=float(gt.sum()/orig.sum()-1),
           fn_total=int(fn.sum()),fn_core=int((fn&orig).sum()),fn_added_boundary=int((fn&annulus).sum()),
           fn_added_boundary_fraction=div((fn&annulus).sum(),fn.sum()),
           fp_total=int(fp.sum()),fp_within2px_fraction=div(sums['fp_near'],fp.sum()),
           centerline_recall=div(sums['skel_hit'],sums['skel_n']),
           thin_centerline_recall=div(sums['thin_hit'],sums['thin_n']),
           mid_centerline_recall=div(sums['mid_hit'],sums['mid_n']),
           thick_centerline_recall=div(sums['thick_hit'],sums['thick_n']),
           oracle_fix_added_boundary_fn_dice=float(base.dice_scores(corrected_boundary,y,.5).mean()),
           oracle_fix_core_fn_dice=float(base.dice_scores(corrected_core,y,.5).mean()),
           oracle_note='Label-assisted upper bounds on validation images only; not achievable test results or proposed submissions.',
           per_image=rows)
    base.save_json(Path(__file__).parent/'bottleneck_audit.json',s)
    print(json.dumps({k:v for k,v in s.items() if k!='per_image'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
