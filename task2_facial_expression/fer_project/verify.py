"""Run after prepare/download; verifies label transfer, finite losses, and branch gradients."""
import json,time
import fer
import numpy as np
import torch

def main():
    fer.seed_everything(42)
    data=np.load(fer.ROOT/'data'/'dataset.npz')
    assert set(data['groups'][data['train_idx']]).isdisjoint(data['groups'][data['val_idx']])
    assert len(data['test_ids'])==7065 and len(set(data['test_ids']))==7065
    indices=np.concatenate([np.flatnonzero(data['labels']==c)[:2] for c in range(6)])
    ds=fer.Faces(data['train_images'][indices],data['labels'][indices],160)
    x=torch.stack([ds[i][0] for i in range(12)]).cuda().to(memory_format=torch.channels_last)
    y=torch.tensor([ds[i][1] for i in range(12)],device='cuda')
    results=[]
    for variant in ['baseline','local','scn','rul']:
        fer.seed_everything(42)
        m=fer.FER(variant).cuda().to(memory_format=torch.channels_last)
        m.train(); t=time.time()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            z,f=m(x); loss=fer.objective(m,z,f,y,2)
        assert z.shape==(12,6) and torch.isfinite(loss)
        loss.backward(); torch.cuda.synchronize()
        gradients={name:float(p.grad.float().norm()) for name,p in m.named_parameters()
                   if p.grad is not None and any(k in name for k in ['local.scale','confidence.weight','log_variance.1.weight'])}
        assert all(np.isfinite(v) and v>0 for v in gradients.values()), gradients
        if variant=='baseline':
            original=torch.load(fer.PRETRAINED,map_location='cpu',weights_only=False)
            head=original.classifier[0] if isinstance(original.classifier,torch.nn.Sequential) else original.classifier
            assert torch.equal(m.classifier.weight.cpu(),head.weight[[0,3,4,6,7,5]])
        results.append({'variant':variant,'loss':loss.item(),'gradients':gradients,'seconds':time.time()-t})
        del m; torch.cuda.empty_cache()
    fer.dump(fer.ROOT/'runs'/'verification.json',results)
    print(json.dumps(results),flush=True)

if __name__=='__main__': main()
