import json, sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
root=Path('/share_datasets/Seal_Model_Reproduction')
sys.path.insert(0,str(root/'NESS-Net-main'))
from infer_seal import load_thermal
resize=transforms.Resize((384,384))
data=root/'NESS-Net-main/dataset_seal_box/train_data'
errors=[]
for p in sorted((root/'Seal_data/train/RGB').glob('*.jpg')):
    stem=p.stem
    image=Image.open(p).convert('RGB')
    thermal=load_thermal(root/f'Seal_data/train/T/{stem}.tif')
    saved=Image.open(data/f'depth/{stem}.png').convert('RGB')
    if not np.array_equal(thermal,saved): errors.append([stem,'thermal'])
    if (data/'img'/p.name).resolve()!=p.resolve(): errors.append([stem,'rgb_source'])
    for folder,im in [('rgb-lab',image),('depth-lab',thermal)]:
        if not np.array_equal(cv2.Canny(np.asarray(resize(im)),10,100),cv2.imread(str(data/f'{folder}/{stem}.png'),0)): errors.append([stem,folder])
    fg=Image.new('L',image.size,0);valid=Image.new('L',image.size,255)
    fd=ImageDraw.Draw(fg);vd=ImageDraw.Draw(valid)
    boxes=[tuple(map(int,l.split())) for l in (root/f'Seal_data/train/box/{stem}.txt').read_text().splitlines()]
    boxes=[(max(0,x1),max(0,y1),min(image.width-1,x2),min(image.height-1,y2)) for x1,y1,x2,y2 in boxes]
    for box in boxes: vd.rectangle(box,fill=0)
    for x1,y1,x2,y2 in boxes:
        cx,cy=(x1+x2)//2,(y1+y2)//2;hx=max(1,(x2-x1+1)//4);hy=max(1,(y2-y1+1)//4)
        center=(max(x1,cx-hx),max(y1,cy-hy),min(x2,cx+hx),min(y2,cy+hy))
        fd.rectangle(center,fill=255);vd.rectangle(center,fill=255)
    for folder,im in [('gt',fg),('mask',valid)]:
        if not np.array_equal(im,Image.open(data/f'{folder}/{stem}.png')):errors.append([stem,folder])
records={}
old=root/'NESS-Net-main/experiments/Seal_box/canny_selected'
for epoch in range(5,36,5):
    directory=old/f'val_epoch_{epoch}_maps';gt=root/'Seal_data/val/GT'
    expected={p.name for p in gt.glob('*.png')}; actual={p.name for p in directory.glob('*.png')}
    bad=[p.name for p in directory.glob('*.png') if cv2.imread(str(p),0).shape!=cv2.imread(str(gt/p.name),0).shape]
    records[str(epoch)]={'count':len(actual),'missing':sorted(expected-actual),'extra':sorted(actual-expected),'bad_size':bad,'log':(old/f'val_epoch_{epoch}_maps_infer.log').read_text()[-150:]}
blank=[]
for p in (root/'results/NESS-Net/Seal/thermal_namlab_edges').glob('*.png'):
    if not np.any(cv2.imread(str(p),0)):blank.append(p.name)
result={'train_errors':errors,'train_count':len(list((data/'img').glob('*.jpg'))),'validation':records,'blank_thermal':blank}
output=root/'results/NESS-Net/Seal/original_eval/input_audit.json'
output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
