"""Fixed, label-independent green-region crops for a seedling scale ablation."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from scipy import ndimage
from plant_bioclip_probe import data_info

def crop_box(image):
    rgb=np.asarray(image,dtype=np.float32)/255.;hsv=np.asarray(image.convert('HSV'))
    r,g,b=rgb.transpose(2,0,1)
    mask=(hsv[:,:,0]>=24)&(hsv[:,:,0]<=130)&(hsv[:,:,1]>=60)&(hsv[:,:,2]>=20)&(2*g-r-b>.08)&(g>1.04*r)&(g>1.02*b)
    mask=ndimage.binary_closing(mask,iterations=1)
    cc,n=ndimage.label(mask);counts=np.bincount(cc.ravel());counts[0]=0
    keep=np.flatnonzero(counts>=max(4,mask.size*.00015));mask=np.isin(cc,keep)
    h,w=mask.shape
    if mask.mean()<.003:return (0,0,w,h),'fallback_low_green'
    yy,xx=np.where(mask);x0,x1=np.quantile(xx,[.01,.99]);y0,y1=np.quantile(yy,[.01,.99])
    side=max((x1-x0+1)*1.25,(y1-y0+1)*1.25,.45*min(w,h))
    cx,cy=(x0+x1)/2,(y0+y1)/2
    left=max(0,min(cx-side/2,w-side));top=max(0,min(cy-side/2,h-side))
    box=(int(left),int(top),min(w,int(np.ceil(left+side))),min(h,int(np.ceil(top+side))))
    return box,'green_bbox'

def main():
    paths,meta=data_info();dest=Path('work/task2_crops_v4');out=Path('outputs/stronger_v4_20260929')
    ids=meta['train_ids']+['test/'+x for x in meta['test_ids']];records=[];previews=[]
    for i,(path,rel) in enumerate(zip(paths,ids)):
        with Image.open(path) as f:image=f.convert('RGB')
        box,method=crop_box(image);crop=image.crop(box);target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True)
        crop.save(target)
        records.append(dict(id=rel,box=box,method=method,area_fraction=(crop.width*crop.height)/(image.width*image.height),
                            original_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),crop_sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
        if i<500 and i%25==0:
            tile=Image.new('RGB',(300,170),'white');tile.paste(ImageOps.contain(image,(145,145)),(0,20));tile.paste(ImageOps.contain(crop,(145,145)),(150,20))
            ImageDraw.Draw(tile).text((4,2),rel.split('/')[1]+' '+str(i),fill='black');previews.append(tile)
    canvas=Image.new('RGB',(1200,170*5),'#dddddd')
    for i,t in enumerate(previews):canvas.paste(t,((i%4)*300,(i//4)*170))
    canvas.save(out/'crop_contact_sheet.jpg')
    info=dict(original_data_hash=meta['data_hash'],classes=meta['classes'],train_ids=meta['train_ids'],test_ids=meta['test_ids'],
              config='HSV24:130 S60 V20 ExG>.08; g>1.04r,g>1.02b; close1; remove tiny components; 1%-99% bounds; 1.25 margin; min_side=.45',records=records)
    (out/'crop_manifest.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(dict(images=len(records),mean_retained_area=float(np.mean([r['area_fraction'] for r in records])),fallback=sum(r['method'].startswith('fallback') for r in records))))

if __name__=='__main__':main()
