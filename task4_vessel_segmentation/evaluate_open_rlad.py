"""Zero-shot RLAD vessel-union evaluation on supplied training images only.

Author preprocessing and weights are preserved. No target fine-tuning occurs.
Source-declared training datasets omit DRIVE; this is not an image-level audit
of the unavailable upstream training corpus.
"""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME',str(ROOT/'open_models/hf_cache'))
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,sys,time
import numpy as np
from PIL import Image
import torch
from safetensors.torch import load_file
import pipeline as base
from operator_confidence import lift
from v3_native import convert_masks

MODEL=ROOT/'open_models/rlad_swinv2_large'
sys.path.insert(0,str(MODEL))
from LUNetv2 import Swinv2Segmenter,Swinv2SegmenterConfig
from image_processing_lunetv2 import Swinv2SegmenterImageProcessor
from image_processing_lunetv2 import MODEL_CONTENT_SIZE,MODEL_PADDED_SIZE,PAD

def load_model():
    config=Swinv2SegmenterConfig.from_pretrained(MODEL,local_files_only=True)
    config.model_pretrained=str(MODEL/'backbone_config')
    model=Swinv2Segmenter(config)
    state=load_file(str(MODEL/'model.safetensors'),device='cpu')
    model.load_state_dict(state,strict=True);del state
    return model.to('cuda').to(torch.bfloat16).eval()

def predict_one(model,processor,image,geometry='unpad'):
    x=processor(image,return_tensors='pt')['pixel_values'].to('cuda',torch.bfloat16)
    if geometry=='unpad':
        # Decoder pixels share the padded input coordinates. Restore the known
        # content rectangle before mapping to the original image, without any
        # label-fitted registration or edits to upstream model weights/code.
        features=model.swinv2(x)[0].transpose(1,2)
        batch,channels,length=features.shape;side=int(length**.5)
        assert side*side==length
        logits=model.decoder(features.reshape(batch,channels,side,side)).float()
        assert logits.shape[-2:]==(MODEL_PADDED_SIZE,MODEL_PADDED_SIZE)
        logits=logits[...,PAD:PAD+MODEL_CONTENT_SIZE,PAD:PAD+MODEL_CONTENT_SIZE]
    else:logits=model(x).float()
    p=torch.sigmoid(logits[:,:2]).amax(1,keepdim=True)
    p=torch.nn.functional.interpolate(p,size=(image.height,image.width),mode='bilinear',align_corners=False)
    return p[0,0].cpu().numpy()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=20)
    ap.add_argument('--geometry',choices=['author','unpad'],default='unpad');args=ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    out=ROOT/'open_models'/('rlad_evaluation_unpadded' if args.geometry=='unpad' else 'rlad_evaluation');out.mkdir(exist_ok=True)
    expected='904c8c20ba38ff23f4350454349ce92e1cb7bbbcefe214243d3f458ecc8f3338'
    assert hashlib.file_digest((MODEL/'model.safetensors').open('rb'),'sha256').hexdigest()==expected
    print('BUILD_MODEL',flush=True)
    model=load_model()
    processor=Swinv2SegmenterImageProcessor()
    _,target,_,ids=base.load_arrays('train')
    predictions=[];timings=[]
    with torch.inference_mode():
        for name in ids[:args.limit]:
            image=Image.open(base.DATA/'train/image'/f'{name}.jpg').convert('RGB')
            torch.cuda.synchronize();start=time.monotonic()
            predictions.append(predict_one(model,processor,image,args.geometry))
            timings.append(time.monotonic()-start)
            print(f'PREDICTED {name} seconds={timings[-1]:.2f} peak_GB={torch.cuda.max_memory_allocated()/1e9:.2f}',flush=True)
    p=np.stack(predictions)
    np.savez_compressed(out/f'native_limit{len(p)}.npz',ids=np.array(ids[:len(p)]),probabilities=p)
    mask=convert_masks(p,.5)
    Image.fromarray(mask[0].astype(np.uint8)*255).save(out/'smoke_vessel_white.png')
    if len(p)!=20:return
    confidence=lift(p)
    old=np.load(ROOT/'v4/selected_probabilities.npz')['oof']
    m=json.loads((ROOT/'v4/metrics.json').read_text(encoding='utf-8'))
    baseline=base.dice_scores(old,target,m['selected']['threshold'])
    results=[]
    for alpha in (0.,.25,.5,.75,1.):
        mixture=alpha*confidence+(1-alpha)*old
        for t in np.round(np.arange(.2,.801,.025),3):
            score=base.dice_scores(mixture,target,float(t))
            results.append(dict(alpha=alpha,threshold=float(t),dice=float(score.mean())))
    best=max(results,key=lambda r:r['dice'])
    family=max((r for r in results if r['alpha']==1.),key=lambda r:r['dice'])
    np.savez_compressed(out/'submission_grid_confidence.npz',ids=np.array(ids),confidence=confidence)
    # mask is already binary on the submission grid.
    fixed=base.dice_scores(mask.astype(np.float32),target,.5)
    result=dict(model='JonathanFhima/RLAD-SwinV2-Large',weights_sha256=expected,
                geometry=args.geometry,unpadded_pixels=PAD if args.geometry=='unpad' else 0,
                zero_shot=True,training_images_used_for_finetuning=0,images=20,
                declared_training_sources_omit_DRIVE=True,upstream_image_level_overlap_audit=False,
                baseline_v4=float(baseline.mean()),fixed_native_threshold_05=float(fixed.mean()),
                calibrated_family=family,calibrated_combination=best,candidates=results,
                inference_seconds=timings,
                note='Target labels used for threshold/mixture selection only. Selection validation is not a new leaderboard result. No Kaggle upload.')
    (out/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('RLAD_EVALUATED '+json.dumps({k:v for k,v in result.items() if k not in ('candidates','inference_seconds')}),flush=True)

if __name__=='__main__':main()
