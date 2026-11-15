"""Nested held-out RBF-SVM comparison using existing frozen DINO features."""
import json
import time
from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from threadpoolctl import threadpool_limits
from plant_v3_features import make_probe, save_submission


def main():
    root=Path('outputs/stronger_20260929')
    source=root/'dino_foreground';out=root/'dino_kernel_probe'
    out.mkdir(exist_ok=True)
    meta=json.loads((source/'feature_metadata.json').read_text())
    z=np.load(source/'features.npz');X=z['train'];Xt=z['test']
    y=np.array(meta['labels']);classes=np.array(meta['classes'])
    model=make_probe().set_params(classifier=SVC(kernel='rbf',probability=True,random_state=42),recipe__views=8)
    grid={'recipe__recipe':['global','foreground','combined'],'classifier__C':[.1,1.,10.,100.],
          'classifier__gamma':['scale']}
    oof=np.zeros((len(y),len(classes)));folds=np.zeros(len(y),int);scores=[]
    with threadpool_limits(2):
        for fold,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y),1):
            begin=time.time()
            search=GridSearchCV(model,grid,cv=StratifiedKFold(4,shuffle=True,random_state=42+fold),n_jobs=1,scoring='accuracy',error_score='raise')
            search.fit(X[tr],y[tr]);assert np.array_equal(search.classes_,classes)
            oof[va]=search.predict_proba(X[va]);folds[va]=fold
            # Submission uses probability argmax, which can differ from SVC.predict.
            score=dict(fold=fold,accuracy=float(accuracy_score(y[va],classes[oof[va].argmax(1)])),params=search.best_params_,seconds=time.time()-begin)
            scores.append(score);print(json.dumps(score),flush=True)
        search=GridSearchCV(model,grid,cv=StratifiedKFold(5,shuffle=True,random_state=47),n_jobs=1,scoring='accuracy',error_score='raise')
        search.fit(X,y);tp=search.predict_proba(Xt)
    report=dict(accuracy=float(accuracy_score(y,classes[oof.argmax(1)])),folds=scores,final_params=search.best_params_,
                confusion_matrix=confusion_matrix(y,classes[oof.argmax(1)],labels=classes).tolist(),
                note='RBF baseline comparison, not a new paper reproduction; outer folds held out throughout fitting and probability calibration.')
    (out/'cv_report.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(out/'probabilities.npz',oof=oof,test=tp,labels=y,classes=classes,folds=folds)
    joblib.dump(dict(estimator=search.best_estimator_,metadata=meta),out/'classifier.joblib')
    save_submission(out/'submission.csv',meta['test_ids'],classes[tp.argmax(1)])
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
