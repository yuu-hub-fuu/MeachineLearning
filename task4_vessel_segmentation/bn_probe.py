"""Train-only BN recalibration diagnostic; validation images never update BN."""
import argparse
import copy
import json
import numpy as np
import torch
from torch import nn
import improve as imp


@torch.inference_mode()
def recalibrate(model, train_images):
    model.train()
    momenta = {}
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            momenta[m] = m.momentum
            m.running_mean.zero_()
            m.running_var.fill_(1)
            m.num_batches_tracked.zero_()
            m.momentum = None
    for arr in np.array_split(train_images, 4):
        x = torch.from_numpy(np.ascontiguousarray(arr)).cuda().contiguous(memory_format=torch.channels_last)
        with torch.autocast('cuda', dtype=torch.float16):
            model(x)
    for m, momentum in momenta.items():
        m.momentum = momentum
    model.eval()


def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    x,y,_,ids=imp.arrays('train')
    oof=np.zeros((20,512,512),np.float32)
    results=[]
    for fold,valid in enumerate(imp.base.fold_splits(ids)):
        train=[i for i in range(20) if i not in valid]
        model=imp.load_model(imp.base.RUN/f'fold{fold}'/'best.pt')
        recalibrate(model,x[train])
        oof[valid]=imp.predict(model,x[valid],'d4')
        results.append(dict(fold=fold,**imp.metrics(oof[valid],y[valid])))
        del model
        torch.cuda.empty_cache()
    summary=dict(all=imp.metrics(oof,y),folds=results)
    imp.base.save_json(imp.V2/'bn_probe.json',summary)
    np.savez_compressed(imp.V2/'bn_probe.npz',ids=np.array(ids),oof=oof.astype(np.float16))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
