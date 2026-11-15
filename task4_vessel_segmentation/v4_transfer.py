"""Generic ImageNet transfer + full-resolution detail stream, native-grid labels.

This is a task-specific model, not a claimed reproduction of DRIU or FR-UNet.
The ImageNet encoder is torchvision ResNet-18; the full-resolution branch and
decoder are independently implemented. No retinal pretraining is used.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18
import pipeline as base
import v3_native as native

ROOT=base.ROOT
WEIGHTS=ROOT/'pretrained/resnet18-f37072fd.pth'
RUN=ROOT/'v4'


class Block(nn.Module):
    def __init__(self,ci,co):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(ci,co,3,padding=1,bias=False),nn.BatchNorm2d(co),nn.ReLU(inplace=True),
                               nn.Conv2d(co,co,3,padding=1,bias=False),nn.BatchNorm2d(co),nn.ReLU(inplace=True))
    def forward(self,x):return self.net(x)


class TransferUNet(nn.Module):
    def __init__(self,pretrained=True,freeze_bn=True):
        super().__init__()
        enc=resnet18(weights=None)
        if pretrained:
            digest=hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
            if digest!='f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec':
                raise RuntimeError('ImageNet weight checksum mismatch')
            enc.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True))
        enc.fc=nn.Identity();enc.avgpool=nn.Identity()
        self.encoder=enc
        self.freeze_bn=freeze_bn
        self.detail=Block(4,32)
        self.dec3=Block(512+256,256)
        self.dec2=Block(256+128,128)
        self.dec1=Block(128+64,64)
        self.dec0=Block(64+64,32)
        self.full=Block(32+32,32)
        self.head=nn.Conv2d(32,1,1)
        self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])

    def train(self,mode=True):
        super().train(mode)
        if mode and self.freeze_bn:
            for m in self.encoder.modules():
                if isinstance(m,nn.BatchNorm2d):m.eval()
        return self

    @staticmethod
    def merge(a,b):
        return torch.cat([F.interpolate(a,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1)

    def forward(self,x):
        detail=self.detail(x)
        e=self.encoder
        a=e.relu(e.bn1(e.conv1((x[:,:3]-self.mean)/self.std)))
        b=e.layer1(e.maxpool(a));c=e.layer2(b);d=e.layer3(c);z=e.layer4(d)
        u=self.dec3(self.merge(z,d));u=self.dec2(self.merge(u,c));u=self.dec1(self.merge(u,b))
        u=self.dec0(self.merge(u,a));u=self.full(self.merge(u,detail))
        return self.head(u)


def factory(args):
    return TransferUNet(pretrained=not args.scratch,freeze_bn=True).to(device='cuda',memory_format=torch.channels_last)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--folds',type=int,nargs='+',default=[2])
    ap.add_argument('--steps',type=int,default=2000);ap.add_argument('--stop-at',type=int,default=800)
    ap.add_argument('--batch',type=int,default=8);ap.add_argument('--patch',type=int,default=160)
    ap.add_argument('--scale',type=int,default=2,help='Recorded for common runtime; does not alter ResNet-18')
    ap.add_argument('--lr',type=float,default=.001)
    ap.add_argument('--seed-offset',type=int,default=0)
    ap.add_argument('--scratch',action='store_true')
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True;torch.use_deterministic_algorithms(False)
    native.RUN=RUN/('scratch' if args.scratch else 'transfer')
    if args.seed_offset:native.RUN=native.RUN/f'seed{args.seed_offset}'
    native.RUN.mkdir(parents=True,exist_ok=True)
    base.save_json(native.RUN/'experiment.json',dict(architecture='TransferUNet with ResNet18 encoder and full-resolution branch',
                   initialization='random' if args.scratch else 'official ImageNet-1K ResNet18, SHA256 verified',
                   encoder_lr_scale=.1,encoder_batchnorm='frozen running stats, affine parameters trainable',
                   target='native binary vessels; submitted-mask operator applied only at validation/export',
                   args=vars(args),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   runtime_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()))
    for fold in args.folds:native.train(args,fold,model_factory=factory)


if __name__=='__main__':main()
