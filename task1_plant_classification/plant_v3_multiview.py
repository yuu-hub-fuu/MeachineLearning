"""Train a classifier on individual orientation views, splitting by source image.

Views from the same image never cross a train/validation boundary. C is divided
by the number of views, preserving the per-image regularization scale.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from threadpoolctl import threadpool_limits
from plant_v3_features import save_submission


class AugmentedProbe(ClassifierMixin,BaseEstimator):
    def __init__(self,recipe='foreground',C=1.):self.recipe=recipe;self.C=C

    def view_features(self,X):
        parts={'global':[0,1,2,3,4],'foreground':[0,1,2,3,5],'combined':[0,1,2,3,4,5]}[self.recipe]
        z=X[:,:,parts,:].copy().reshape(len(X)*X.shape[1],-1)
        return z/np.linalg.norm(z,axis=1,keepdims=True).clip(1e-8)

    def fit(self,X,y):
        self.classes_=np.unique(y)
        self.estimator_=make_pipeline(StandardScaler(),LogisticRegression(C=self.C/X.shape[1],max_iter=1500))
        self.estimator_.fit(self.view_features(X),np.repeat(y,X.shape[1]))
        return self

    def predict_proba(self,X):
        p=self.estimator_.predict_proba(self.view_features(X))
        return p.reshape(len(X),X.shape[1],-1).mean(1)

    def predict(self,X):return self.classes_[self.predict_proba(X).argmax(1)]


def main():
    a=argparse.ArgumentParser();a.add_argument('--cache-dir',default='outputs/stronger_20260929/dino_foreground')
    a.add_argument('--run-dir',default='outputs/stronger_20260929/dino_augmented_probe');args=a.parse_args()
    cache=Path(args.cache_dir);out=Path(args.run_dir);out.mkdir(exist_ok=True,parents=True)
    meta=json.loads((cache/'feature_metadata.json').read_text());data=np.load(cache/'features.npz')
    X,Xt=data['train'],data['test'];y=np.array(meta['labels']);classes=np.array(meta['classes'])
    from plant_v3_multiview import AugmentedProbe as SerializableProbe
    grid=dict(recipe=['global','foreground','combined'],C=[.1,1.,10.])
    oof=np.zeros((len(y),len(classes)));folds=np.zeros(len(y),int);details=[]
    with threadpool_limits(2):
        for fold,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y),1):
            search=GridSearchCV(SerializableProbe(),grid,cv=StratifiedKFold(4,shuffle=True,random_state=42+fold),n_jobs=1,error_score='raise',scoring='accuracy')
            search.fit(X[tr],y[tr]);oof[va]=search.predict_proba(X[va]);folds[va]=fold
            d=dict(fold=fold,accuracy=float(accuracy_score(y[va],classes[oof[va].argmax(1)])),params=search.best_params_)
            details.append(d);print(json.dumps(d),flush=True)
        final=GridSearchCV(SerializableProbe(),grid,cv=StratifiedKFold(5,shuffle=True,random_state=47),n_jobs=1,error_score='raise',scoring='accuracy')
        final.fit(X,y);test=final.predict_proba(Xt)
    report=dict(accuracy=float(accuracy_score(y,classes[oof.argmax(1)])),folds=details,
                fold_sd=float(np.std([d['accuracy'] for d in details],ddof=1)),final_params=final.best_params_,
                confusion_matrix=confusion_matrix(y,classes[oof.argmax(1)],labels=classes).tolist(),
                classification_report=classification_report(y,classes[oof.argmax(1)],output_dict=True))
    (out/'cv_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (out/'feature_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    np.savez_compressed(out/'probabilities.npz',oof=oof,test=test,labels=y,classes=classes,folds=folds)
    joblib.dump(dict(estimator=final.best_estimator_,metadata=meta),out/'classifier.joblib')
    save_submission(out/'submission.csv',meta['test_ids'],classes[test.argmax(1)])
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
