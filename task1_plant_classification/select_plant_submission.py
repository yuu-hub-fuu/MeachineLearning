"""Compare completed OOF experiments and fixed equal-weight blends.

This comparison reuses validation data; the selected score is an exploratory
development estimate, not a fresh independent test score or a leaderboard score.
"""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,log_loss,confusion_matrix,classification_report
from plant_v3_features import save_submission


def main():
    root=Path('outputs/stronger_20260929')
    meta=json.loads((root/'dino_foreground/feature_metadata.json').read_text())
    labels=np.array(meta['labels']);classes=np.array(meta['classes']);models={}
    folds=None
    for name in ['dino_foreground','dino_augmented_probe','dino_kernel_probe','dino_lpft']:
        file=root/name/'probabilities.npz'
        if not file.exists(): raise RuntimeError(f'Experiment not finished: {name}')
        z=np.load(file)
        assert np.array_equal(z['labels'],labels) and np.array_equal(z['classes'],classes)
        assert z['oof'].shape==(500,5) and z['test'].shape==(378,5)
        if folds is None:folds=z['folds']
        else:assert np.array_equal(folds,z['folds'])
        for key in ['oof','test']:
            assert np.isfinite(z[key]).all() and (z[key]>=0).all() and np.allclose(z[key].sum(1),1,atol=1e-5)
        # AMP softmax and view averaging leave tiny float32 row-sum errors.
        # Renormalize in float64 before log-loss and mixed-model arithmetic.
        models[name]={key:z[key].astype(np.float64)/z[key].sum(1,keepdims=True,dtype=np.float64)
                      for key in ['oof','test']}
    # Verify actual data order against the independently recorded training hash.
    config=json.loads((root/'dino_lpft/config.json').read_text())
    data_root=Path(config['train_dir']).parent
    paths=[data_root/x for x in meta['train_ids']]+[Path(config['test_dir'])/x for x in meta['test_ids']]
    h=hashlib.sha256()
    for path in paths:h.update(path.name.encode());h.update(hashlib.sha256(path.read_bytes()).digest())
    assert h.hexdigest()==config['data_hash'], 'Data/order mismatch between feature and fine-tuning runs'
    recipes={name:{name:1.} for name in models}
    # Fixed blends, specified without fitting weights or looking at test labels.
    recipes['foreground_plus_lpft']={'dino_foreground':.5,'dino_lpft':.5}
    recipes['augmented_plus_lpft']={'dino_augmented_probe':.5,'dino_lpft':.5}
    recipes['kernel_plus_lpft']={'dino_kernel_probe':.5,'dino_lpft':.5}
    candidates={};scores=[]
    for name,weights in recipes.items():
        oof=sum(models[m]['oof']*w for m,w in weights.items())
        test=sum(models[m]['test']*w for m,w in weights.items())
        pred=classes[oof.argmax(1)]
        fold_scores=[float(accuracy_score(labels[folds==f],pred[folds==f])) for f in range(1,6)]
        score=dict(name=name,accuracy=float(accuracy_score(labels,pred)),log_loss=float(log_loss(labels,oof,labels=classes)),
                   fold_scores=fold_scores,fold_sd=float(np.std(fold_scores,ddof=1)),weights=weights)
        scores.append(score);candidates[name]=dict(oof=oof,test=test)
    scores.sort(key=lambda s:(-s['accuracy'],s['log_loss']))
    chosen=scores[0];best=candidates[chosen['name']];pred=classes[best['test'].argmax(1)]
    target=root/'submission_v3_20260929.csv';save_submission(target,meta['test_ids'],pred)
    with (root/'baseline_vitl_submission.csv').open(encoding='utf-8-sig') as f:old=list(csv.DictReader(f))
    assert [r['ID'] for r in old]==meta['test_ids']
    changed=int(sum(p!=r['Category'] for p,r in zip(pred,old)))
    reference=np.load(root/'dino_foreground/probabilities.npz')['baseline_oof']
    old_correct=classes[reference.argmax(1)]==labels
    new_correct=classes[best['oof'].argmax(1)]==labels
    paired=dict(reference='old feature recipe, fixed C=10 refit on identical five folds',
                reference_accuracy=float(old_correct.mean()),
                corrected=int((~old_correct & new_correct).sum()),
                newly_wrong=int((old_correct & ~new_correct).sum()),
                both_wrong=int((~old_correct & ~new_correct).sum()))
    report=dict(selected=chosen,candidates=scores,changed_test_predictions=changed,
                paired_reference=paired,
                rows=len(pred),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                confusion_matrix=confusion_matrix(labels,classes[best['oof'].argmax(1)],labels=classes).tolist(),
                classification_report=classification_report(labels,classes[best['oof'].argmax(1)],output_dict=True),
                online_score=None,note='Selected using reused OOF development estimates; new public score unknown. CNN was stopped after two unsuccessful folds and excluded.')
    (root/'selection_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    np.savez_compressed(root/'selected_probabilities.npz',oof=best['oof'],test=best['test'],classes=classes,labels=labels,folds=folds)
    with (root/'selected_oof.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['ID','true','pred','fold']+list(classes))
        for i in range(len(labels)):w.writerow([meta['train_ids'][i],labels[i],classes[best['oof'][i].argmax()],folds[i]]+best['oof'][i].tolist())
    with (root/'prediction_review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['ID','Category','max_score','old_Category','changed'])
        for i,prediction in enumerate(pred):w.writerow([meta['test_ids'][i],prediction,float(best['test'][i].max()),old[i]['Category'],prediction!=old[i]['Category']])
    with target.open(encoding='utf-8-sig') as f:check=list(csv.DictReader(f))
    assert len(check)==378 and [r['ID'] for r in check]==meta['test_ids']
    assert len({r['ID'] for r in check})==378 and all(r['Category'] in classes for r in check)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
