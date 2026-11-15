"""Small CPU end-to-end check of feature fitting and full-refit serialization."""
from upgrade_experiments import *


def main():
    seed_everything(42);cache=ROOT/'diagnostics'/'upgrade_features';cache.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(42);n=72;dim=16;y=np.arange(n)%6
    x=rng.normal(size=(n,2,dim)).astype(np.float16)*.1
    for i in range(n):x[i,:,y[i]]+=1
    weight=np.zeros((6,dim),np.float32);weight[:,:6]=np.eye(6)*2
    np.save(cache/'train_features.npy',x);np.save(cache/'test_features.npy',x[:12])
    np.save(cache/'train_probs.npy',np.full((n,2,6),1/6,dtype=np.float32))
    np.savez(cache/'head.npz',weight=weight,bias=np.zeros(6,dtype=np.float32))
    packed={'labels':y,'train_idx':np.arange(60),'val_idx':np.arange(60,72),'test_ids':np.array([f'fake_{i}' for i in range(12)])}
    checks=[]
    for mode in ['linear','adapter_ce','supcon']:
        p,epoch,path=train_head('diagnostic',mode,cache,packed,epochs=3,device='cpu')
        assert p.shape==(12,6) and np.allclose(p.sum(1),1,atol=1e-6)
        assert (p.argmax(1)==y[60:]).all()
        p,epoch,path=train_head('diagnostic',mode,cache,packed,epochs=3,full=True,chosen_epoch=2,device='cpu')
        saved=torch.load(path,map_location='cpu',weights_only=True)
        model=FeatureHead(torch.from_numpy(weight),torch.zeros(6),mode);model.load_state_dict(saved['model'])
        reloaded=predict_head(model,x[:12],np.arange(12),device='cpu')
        assert np.array_equal(p,reloaded) and saved['best_epoch']==2
        assert not torch.equal(model.fc.weight,torch.from_numpy(weight))
        checks.append({'mode':mode,'full_refit_epochs':2,'checkpoint_reproduces_probabilities':True,'optimizer_updated_head':True})
    dump(OUT/'upgrade_training_verification.json',{'passed':True,'checks':checks,'synthetic_diagnostic_only':True})


if __name__=='__main__':main()
