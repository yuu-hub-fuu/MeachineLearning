"""Compare v4 candidates against the user-reported 0.9735 v3 submission.

The candidate family and equal blending weights are specified before all v4
results are available. Reused OOF selection is exploratory, not a public score.
"""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,log_loss,confusion_matrix,classification_report
from plant_v3_features import save_submission
from plant_bioclip_probe import data_info

ROOT=Path('outputs/stronger_v4_20260929')
OLD=Path('outputs/stronger_20260929')

def main():
    meta=json.loads((OLD/'dino_foreground/feature_metadata.json').read_text())
    _,current=data_info()
    for key in ('train_ids','test_ids','labels','classes'):assert current[key]==meta[key]
    data_hash=current['data_hash']
    assert json.loads((ROOT/'bioclip_lpft/config.json').read_text())['data_hash']==data_hash
    assert json.loads((ROOT/'dino_518/inference_config.json').read_text())['training_config']['data_hash']==data_hash
    assert json.loads((ROOT/'crop_manifest.json').read_text())['original_data_hash']==data_hash
    models={};labels=np.array(meta['labels']);classes=np.array(meta['classes']);folds=None
    sources={'baseline336':OLD/'dino_lpft','resolution518':ROOT/'dino_518',
             'plant_crop':ROOT/'dino_crop','bioclip25':ROOT/'bioclip25',
             'bioclip_lpft':ROOT/'bioclip_lpft','kernel':ROOT/'kernel',
             'graph_a2':ROOT/'graph_a2','graph_a5':ROOT/'graph_a5'}
    for name,path in sources.items():
        z=np.load(path/'probabilities.npz')
        assert np.array_equal(z['labels'],labels) and np.array_equal(z['classes'],classes)
        if folds is None:folds=z['folds']
        else:assert np.array_equal(folds,z['folds'])
        if name in ('plant_crop','bioclip25','kernel','graph_a2','graph_a5'):
            m=json.loads((path/'feature_metadata.json').read_text())
            assert m['train_ids']==meta['train_ids'] and m['test_ids']==meta['test_ids']
            if name in ('bioclip25','kernel','graph_a2','graph_a5'):assert m['data_hash']==data_hash
        models[name]={}
        for key,n in [('oof',500),('test',378)]:
            p=z[key].astype(np.float64)
            assert p.shape==(n,5) and np.isfinite(p).all() and (p>=0).all()
            assert np.allclose(p.sum(1),1,atol=1e-5)
            models[name][key]=p/p.sum(1,keepdims=True)
    baseline_path=OLD/'submission_v3_20260929.csv'
    assert hashlib.sha256(baseline_path.read_bytes()).hexdigest()=='88e0830d1743da1cf3f528e6437525b886e903da9905a39fe24a9e53a040a882'
    with baseline_path.open(encoding='utf-8-sig') as f:old=list(csv.DictReader(f))
    assert [r['ID'] for r in old]==meta['test_ids']
    assert np.array_equal(classes[models['baseline336']['test'].argmax(1)],[r['Category'] for r in old])
    recipes={name:{name:1.} for name in models}
    for name in ('resolution518','plant_crop','bioclip25','bioclip_lpft','kernel'):
        recipes['baseline_plus_'+name]={'baseline336':.5,name:.5}
    recipes['baseline_crop_bio']={'baseline336':1/3,'plant_crop':1/3,'bioclip25':1/3}
    recipes['baseline_crop_bio_lpft']={'baseline336':1/3,'plant_crop':1/3,'bioclip_lpft':1/3}
    candidates={};scores=[]
    for name,weights in recipes.items():
        d={key:sum(models[k][key]*w for k,w in weights.items()) for key in ('oof','test')}
        pred=classes[d['oof'].argmax(1)]
        row=dict(name=name,accuracy=float(accuracy_score(labels,pred)),log_loss=float(log_loss(labels,d['oof'],labels=classes)),
                 folds=[float(accuracy_score(labels[folds==i],pred[folds==i])) for i in range(1,6)],weights=weights)
        scores.append(row);candidates[name]=d
    scores.sort(key=lambda r:(-r['accuracy'],r['log_loss']))
    chosen=scores[0];d=candidates[chosen['name']];pred=classes[d['test'].argmax(1)]
    target=ROOT/'submission_v4_20260929.csv';save_submission(target,meta['test_ids'],pred)
    old_ok=classes[models['baseline336']['oof'].argmax(1)]==labels;new_ok=classes[d['oof'].argmax(1)]==labels
    r=dict(selected=chosen,candidates=scores,baseline_user_reported_public_score=.9735,new_public_score=None,
           changed_test_predictions=int(np.sum(pred!=np.array([r['Category'] for r in old]))),
           paired_oof=dict(corrected=int((~old_ok&new_ok).sum()),newly_wrong=int((old_ok&~new_ok).sum()),both_wrong=int((~old_ok&~new_ok).sum())),
           confusion_matrix=confusion_matrix(labels,classes[d['oof'].argmax(1)],labels=classes).tolist(),
           classification_report=classification_report(labels,classes[d['oof'].argmax(1)],output_dict=True),
           submission_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
           note='Reused OOF candidate selection; not an independent evaluation or a known new public score.')
    (ROOT/'selection_report.json').write_text(json.dumps(r,indent=2))
    np.savez_compressed(ROOT/'selected_probabilities.npz',**d,classes=classes,labels=labels,folds=folds)
    with (ROOT/'oof_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['ID','true','fold','v3_Category','v4_Category','v3_correct','v4_correct'])
        w.writerows((i,y,int(fold),old_p,new_p,bool(ok0),bool(ok1))
                    for i,y,fold,old_p,new_p,ok0,ok1 in zip(meta['train_ids'],labels,folds,
                    classes[models['baseline336']['oof'].argmax(1)],classes[d['oof'].argmax(1)],old_ok,new_ok))
    with (ROOT/'prediction_changes.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['ID','v3_Category','v4_Category','changed','v3_max_probability','v4_max_probability']+list(classes))
        w.writerows([i,prev['Category'],p,prev['Category']!=p,float(models['baseline336']['test'][j].max()),float(d['test'][j].max())]+d['test'][j].tolist()
                    for j,(i,prev,p) in enumerate(zip(meta['test_ids'],old,pred)))
    with target.open(encoding='utf-8-sig') as f:check=list(csv.DictReader(f))
    assert len(check)==378 and [r['ID'] for r in check]==meta['test_ids'] and len(set(meta['test_ids']))==378
    assert all(r['Category'] in classes for r in check)
    print(json.dumps(r,indent=2),flush=True)

if __name__=='__main__':main()
