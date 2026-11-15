"""LP-FT adaptation of the last two BioCLIP 2.5 vision blocks on an 8GB GPU.

The pretrained prefix and projection stay frozen in half precision; all trainable
parameters are float32 so PyTorch GradScaler can safely unscale their gradients.
"""
import json
import sys
import torch
from torch import nn
import torch.nn.functional as F
import plant_finetune as runner
from plant_bioclip_probe import load_visual,CACHE

class BioFeatures(nn.Module):
    def __init__(self,device):
        super().__init__();self.visual,_=load_visual(device)
    def forward(self,x):
        out=self.visual.forward_intermediates(x,indices=4,normalize_intermediates=True,
                                             output_fmt='NLC',output_extra_tokens=True)
        patch=out['image_intermediates'][-1].float();side=int(patch.shape[1]**.5)
        rgb=x.float()*x.new_tensor(runner.STD).view(1,3,1,1)+x.new_tensor(runner.MEAN).view(1,3,1,1)
        r,g,b=rgb.unbind(1);green=F.relu(2*g-r-b-.035)*((g>r)&(g>b))
        w=F.adaptive_avg_pool2d(green[:,None],(side,side)).flatten(1)
        w=torch.where(w.sum(1,keepdim=True)>.05,w,torch.ones_like(w))
        fg=(patch*w[...,None]).sum(1)/w.sum(1,keepdim=True)
        parts=[out['image_features'].float()]+[c[:,0].float() for c in out['image_intermediates_prefix']]+[patch.mean(1),fg]
        return F.normalize(torch.cat([F.normalize(p,dim=1) for p in parts],dim=1),dim=1)

class BioNetwork(nn.Module):
    def __init__(self,device):
        super().__init__();self.features=BioFeatures(device);self.classifier=nn.Sequential(nn.Identity())
    def forward(self,x):return self.classifier(self.features(x))
    def requires_grad_(self,requires_grad=True):
        super().requires_grad_(requires_grad)
        if requires_grad:
            self.features.visual.requires_grad_(False)
            for block in self.features.visual.transformer.resblocks[-2:]:block.float().requires_grad_(True)
            self.features.visual.ln_post.float().requires_grad_(True)
            for name,p in self.named_parameters():
                if p.requires_grad and p.dtype!=torch.float32:raise ValueError(f'Trainable FP16 parameter: {name}')
        return self

def make_model(device):return BioNetwork(device).eval()

if __name__=='__main__':
    cfg=json.loads((CACHE/'open_clip_config.json').read_text())
    runner.MEAN=tuple(cfg['preprocess_cfg']['mean']);runner.STD=tuple(cfg['preprocess_cfg']['std'])
    runner.features_model=make_model;runner.__file__=__file__
    defaults={'--run-dir':'outputs/stronger_v4_20260929/bioclip_lpft','--size':'224','--epochs':'15',
              '--batch-size':'4','--backbone-lr':'0.00001','--head-lr':'0.0001'}
    for key,value in defaults.items():
        if key not in sys.argv:sys.argv.extend([key,value])
    runner.main()
