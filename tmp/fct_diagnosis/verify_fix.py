import json
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

root=Path('/share_datasets/Seal_Model_Reproduction')
sys.path.insert(0,str(root/'FCT-SAM-WSCOD-main'))
from prepare_seal import choose_consistent_mask,label_response_filter,agrees_with_prompts,check_pseudo_labels
out=root/'results/FCT-SAM-WSCOD/diagnosis_20261103'
records=json.loads((out/'sam_audit.json').read_text());verified=[];geometry=[]
for record in records:
    name=record['name'];z=np.load(out/f'sam_candidates_{name}.npz');votes=[]
    height,width=z['masks'].shape[-2:]
    for index in range(10):
        selected=choose_consistent_mask(z['masks'][index],z['scores'][index],z['xy'],z['labels'])
        if selected is None:selected=choose_consistent_mask(z['direct_masks'][index],z['direct_scores'][index],z['xy'],z['labels'])
        if selected is not None:votes.append(selected)
        view=record['views'][index];xy=z['xy']*np.array([768/width,768/height]);coords=xy.copy()
        for _ in range(view['rotation']):coords=np.stack((coords[:,1],767-coords[:,0]),axis=1)
        if view['flip']:coords[:,0]=767-coords[:,0]
        coords*=view['side']/768;coords/=view['side']/768
        if view['flip']:coords[:,0]=767-coords[:,0]
        for _ in range(view['rotation']):coords=np.stack((767-coords[:,1],coords[:,0]),axis=1)
        coords/=np.array([768/width,768/height]);geometry.append(float(np.max(np.abs(coords-z['xy']))))
    result=label_response_filter(votes)
    assert result is not None and agrees_with_prompts(result,z['xy'],z['labels'])
    prior=cv2.imread(str(out/f'consistent_preview_{name}.png'),0)>128
    assert np.array_equal(result,prior)
    verified.append({'name':name,'accepted_views':len(votes),'area':float(result.mean())})
prompts=json.loads((root/'Seal_data/train/SAM_1fg5bg/prompts.json').read_text())
invalid=check_pseudo_labels(sorted((root/'Seal_data/train/RGB').glob('*')),root/'Seal_data/train/SAM_1fg5bg',prompts)
report={'fixed_candidate_replay_images':len(verified),'all_final_prompts_satisfied':True,
    'coordinate_roundtrip_max_pixels':max(geometry),'mean_fixed_area':float(np.mean([x['area'] for x in verified])),
    'existing_label_invalid_count':len(invalid),'total_existing':1500,
    'training_started':False,'quality_status':'prompt consistency passed on sample; large-area false foreground remains; not cleared for long training',
    'images':verified}
(out/'fix_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
canvas=Image.new('RGB',(1200,3*230),(255,255,255));draw=ImageDraw.Draw(canvas)
for row,name in enumerate(['1','1268','728']):
    image_path=next((root/'Seal_data/train/RGB').glob(name+'.*'))
    assets=[Image.open(image_path).convert('RGB'),Image.open(root/'Seal_data/train/GT'/f'{name}.png').convert('RGB'),
        Image.open(root/'Seal_data/train/SAM_1fg5bg'/f'{name}.png').convert('RGB'),Image.open(out/f'consistent_preview_{name}.png').convert('RGB')]
    for col,(image,label) in enumerate(zip(assets,['RGB','GT (diagnostic only)','Existing pseudo','Prompt-consistent candidate'])):
        image.thumbnail((296,200));canvas.paste(image,(col*300,row*230+25));draw.text((col*300+4,row*230+5),f'{name}: {label}',fill='black')
canvas.save(out/'pseudo_examples.png')
print(json.dumps(report,indent=2))
