"""Author checkpoints with explicit preprocessing and six-class mapping.

QCS keeps the exact single-image inference branch learned by the author's
four-branch training. It is not a reimplementation of QCS training.
"""
from fer import ROOT, torch, nn, np, F, Image, T
from torchvision.models import resnet18
import json

SPECS={
 'nla':{'filename':'nla_rafdb_best.pth','order':[5,1,3,4,0,6], 'size':224,'normalize':True,
        'mean':[.577,.4494,.4001],'std':[.2628,.2395,.2383],
        'source_classes':['Surprise','Fear','Disgust','Happy','Sad','Angry','Neutral']},
 'qcs':{'filename':'qcs_ferplus_best.pth','order':[4,6,1,3,2,0], 'size':224,'resize':232,'normalize':True,
        'source_classes':['Neutral','Happy','Surprise','Sad','Angry','Disgust','Fear','Contempt']},
 'rmn':{'filename':'resmasking_fer2013.pth','order':[0,2,3,4,5,6], 'size':224,'normalize':False,
        'source_classes':['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']},
}
for _kind in ['hf_dima','hf_mo','hf_trpakov']:
    SPECS[_kind]={'filename':f'{_kind}/model.safetensors','size':224,'normalize':True,
                 'order':None,'source_classes':None,'source_image_overlap_possible':True}


class NLANetwork(nn.Module):
    def __init__(self):
        super().__init__(); backbone=resnet18(weights=None)
        self.features=nn.Sequential(*list(backbone.children())[:-2])
        self.features2=nn.Sequential(*list(backbone.children())[-2:-1])
        self.fc=nn.Linear(512,7);self.embed=nn.Linear(512,1024)
    def forward(self,x):return self.fc(self.features2(self.features(x)).flatten(1))


class QCSInference(nn.Module):
    def __init__(self):
        super().__init__()
        from vendor.qcs.ir50 import Backbone
        from vendor.qcs.vit_model_8cls import VisionTransformer,PatchEmbed
        self.VIT_base=VisionTransformer(depth=2,drop_ratio=0,embed_dim=768)
        self.ir_back=Backbone(50,0.,'ir')
        self.conv1=nn.Conv2d(64,64,3,2,1);self.conv2=nn.Conv2d(128,128,3,2,1);self.conv3=nn.Conv2d(256,256,3,2,1)
        self.embed_q=nn.Sequential(nn.Conv2d(64,768,3,2,1),nn.Conv2d(768,768,3,2,1))
        self.embed_k=nn.Sequential(nn.Conv2d(128,768,3,2,1))
        self.embed_v=PatchEmbed(img_size=14,patch_size=14,in_c=256,embed_dim=768)
    def forward(self,x):
        a,b,c=self.ir_back(x)
        a=self.embed_q(self.conv1(a)).flatten(2).transpose(1,2)
        b=self.embed_k(self.conv2(b)).flatten(2).transpose(1,2)
        c=self.embed_v(self.conv3(c))
        logits,_=self.VIT_base(torch.cat([a,b,c],dim=1))
        return logits


class UpgradeFER(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind;spec=SPECS[kind].copy()
        if kind.startswith('hf_'):
            from transformers import ViTForImageClassification
            folder=ROOT/'pretrained'/kind
            config=json.loads((folder/'config.json').read_text())
            classes=[config['id2label'][str(i)].lower() for i in range(len(config['id2label']))]
            aliases={'anger':'angry','happiness':'happy','sadness':'sad','surprised':'surprise','fearful':'fear'}
            classes=[aliases.get(c,c) for c in classes]
            spec['order']=[classes.index(c) for c in ['angry','fear','happy','sad','surprise','neutral']]
            spec['source_classes']=classes
            self.source_model,info=ViTForImageClassification.from_pretrained(str(folder),local_files_only=True,
                use_safetensors=True,output_loading_info=True)
            if any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):raise RuntimeError(info)
            head=self.source_model.classifier
            self._set_head(head,spec,len(self.source_model.state_dict()),0)
            return
        checkpoint=torch.load(ROOT/'pretrained'/spec['filename'],map_location='cpu',weights_only=True)
        state=next((checkpoint[k] for k in ['state_dict','net','model_state_dict','model']
                    if isinstance(checkpoint,dict) and k in checkpoint and isinstance(checkpoint[k],dict)),checkpoint)
        state={k.removeprefix('module.'):v for k,v in state.items()}
        if kind=='nla':
            self.source_model=NLANetwork();head=self.source_model.fc
        elif kind=='qcs':
            self.source_model=QCSInference();head=self.source_model.VIT_base.head.linear
            state={k.replace('embed_1.','embed_q.').replace('embed_2.','embed_k.').replace('embed_3.','embed_v.'):v for k,v in state.items()}
        else:
            from vendor.rmn.resmasking import resmasking_dropout1
            with torch.device('meta'):self.source_model=resmasking_dropout1()
            head=self.source_model.fc[1]
        expected=set(self.source_model.state_dict());missing=expected-set(state);extra=set(state)-expected
        allowed_extra=('VIT_cross.','cross_attention_1.','cross_attention_2.','cross_attention_3.') if kind=='qcs' else ()
        bad_extra=[key for key in extra if not key.startswith(allowed_extra)]
        if missing or bad_extra:raise RuntimeError({'kind':kind,'missing':sorted(missing),'unexpected':sorted(bad_extra)[:30]})
        self.source_model.load_state_dict({k:v for k,v in state.items() if k in expected},strict=True,assign=(kind=='rmn'))
        self._set_head(head,spec,len(expected),len(extra))
    def _set_head(self,head,spec,count,excluded):
        self.register_buffer('order',torch.tensor(spec['order'],dtype=torch.long))
        self.feature=None
        head.register_forward_pre_hook(self._capture)
        self.dim=head.in_features
        self.initial_weight=head.weight.detach()[spec['order']].clone()
        self.initial_bias=head.bias.detach()[spec['order']].clone()
        self.load_report={'kind':self.kind,'loaded_tensors':count,'excluded_training_only_tensors':excluded,
                          'source_class_order':spec['source_classes'],'selected_rows':spec['order']}
    def _capture(self,module,args):self.feature=args[0]
    def forward(self,x):
        native=self.source_model(x)
        if self.kind.startswith('hf_'):native=native.logits
        return native[:,self.order],self.feature


class UpgradeImages(torch.utils.data.Dataset):
    def __init__(self,images,kind,crop=0):
        self.images=images;self.crop=crop;self.kind=kind
        spec=SPECS[kind]
        interpolation=T.InterpolationMode.BILINEAR
        if kind.startswith('hf_'):
            spec=spec.copy();cfg=json.loads((ROOT/'pretrained'/kind/'preprocessor_config.json').read_text())
            size=cfg.get('size',224)
            if isinstance(size,dict):
                assert size.get('height',224)==size.get('width',224)
                size=size.get('height',size.get('shortest_edge',224))
            spec.update(size=size,mean=cfg.get('image_mean',[.5]*3),std=cfg.get('image_std',[.5]*3),normalize=cfg.get('do_normalize',True))
            assert cfg.get('do_resize',True) and cfg.get('do_rescale',True)
            assert abs(cfg.get('rescale_factor',1/255)-1/255)<1e-8
            interpolation={2:T.InterpolationMode.BILINEAR,3:T.InterpolationMode.BICUBIC}[cfg.get('resample',2)]
        size=spec.get('resize',spec['size'])
        ops=[] if kind=='rmn' else [T.Resize((size,size),interpolation=interpolation)]
        if size!=spec['size']:ops.append(T.CenterCrop(spec['size']))
        ops.append(T.ToTensor())
        if spec['normalize']:ops.append(T.Normalize(spec.get('mean',[.485,.456,.406]),spec.get('std',[.229,.224,.225])))
        self.transform=T.Compose(ops)
    def __len__(self):return len(self.images)
    def __getitem__(self,i):
        img=Image.fromarray(self.images[i]).convert('RGB')
        if self.crop:img=img.crop((self.crop,self.crop,48-self.crop,48-self.crop))
        if self.kind=='rmn':
            import sys
            if str(ROOT/'dependencies') not in sys.path:sys.path.insert(0,str(ROOT/'dependencies'))
            import cv2
            img=Image.fromarray(cv2.resize(np.asarray(img),(224,224),interpolation=cv2.INTER_LINEAR))
        return self.transform(img),i
