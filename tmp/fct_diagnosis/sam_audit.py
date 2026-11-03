import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

root=Path('/share_datasets/Seal_Model_Reproduction')
out=root/'results/FCT-SAM-WSCOD/diagnosis_20261103'
out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)
sam=sam_model_registry['vit_h'](checkpoint=str(root/'FCT-SAM-WSCOD-main/pretrained/sam_vit_h_4b8939.pth')).cuda().eval()
predictor=SamPredictor(sam)
prompts=json.loads((root/'Seal_data/train/SAM_1fg5bg/prompts.json').read_text())
files=sorted((root/'Seal_data/train/RGB').glob('*'))
indices=np.linspace(0,len(files)-1,16,dtype=int)
reports=[]

def sample_masks(masks,points):
    xy=np.rint(points).astype(int)
    xy[:,0]=np.clip(xy[:,0],0,masks.shape[-1]-1)
    xy[:,1]=np.clip(xy[:,1],0,masks.shape[-2]-1)
    return masks[:,xy[:,1],xy[:,0]]

def restore(masks,rotation,flip,width,height):
    result=[]
    for mask in masks:
        m=cv2.resize(mask.astype(np.uint8),(768,768),interpolation=cv2.INTER_NEAREST)
        if flip:m=np.flip(m,axis=1)
        m=np.rot90(m,-rotation)
        result.append(cv2.resize(m,(width,height),interpolation=cv2.INTER_NEAREST)>0)
    return np.stack(result)

with torch.inference_mode():
    for index in indices:
        path=files[index];image=np.asarray(Image.open(path).convert('RGB'));height,width=image.shape[:2]
        base=cv2.resize(image,(768,768),interpolation=cv2.INTER_LINEAR)
        original_xy=np.asarray(prompts[path.stem]['xy'],dtype=np.float32)
        xy=original_xy*np.array([768/width,768/height],dtype=np.float32)
        labels=np.asarray(prompts[path.stem]['labels']);rng=np.random.default_rng(1042+int(index))
        all_masks=[];all_scores=[];before=[];after=[];views=[];fallback_masks=[];fallback_scores=[]
        for iteration in range(10):
            rotation=int(rng.integers(4));flip=bool(rng.integers(2));side=int(768*rng.choice([.5,1.,2.]))
            augmented=np.rot90(base,rotation);coords=xy.copy()
            for _ in range(rotation):coords=np.stack((coords[:,1],767-coords[:,0]),axis=1)
            if flip:augmented=np.flip(augmented,axis=1);coords[:,0]=767-coords[:,0]
            augmented=cv2.resize(augmented,(side,side),interpolation=cv2.INTER_LINEAR);coords*=side/768
            predictor.set_image(np.ascontiguousarray(augmented))
            _,first_scores,logits=predictor.predict(point_coords=coords[:1],point_labels=labels[:1],multimask_output=True)
            prior=logits[np.argmax(first_scores)][None]
            masks,scores,_=predictor.predict(point_coords=coords,point_labels=labels,mask_input=prior,multimask_output=True)
            initial_consistency=sample_masks(masks,coords)==labels
            restored=restore(masks,rotation,flip,width,height)
            restored_consistency=sample_masks(restored,original_xy)==labels
            # A direct all-point pass tests whether a failed positive-only prior causes the inconsistency.
            direct,ds,_=predictor.predict(point_coords=coords,point_labels=labels,multimask_output=True)
            direct=restore(direct,rotation,flip,width,height)
            all_masks.append(restored);all_scores.append(scores);before.append(initial_consistency);after.append(restored_consistency)
            fallback_masks.append(direct);fallback_scores.append(ds)
            views.append({'rotation':rotation,'flip':flip,'side':side,'prior_shape':list(prior.shape),
                'prior_min':float(prior.min()),'prior_max':float(prior.max()),'scores':scores.tolist(),
                'selected':int(np.argmax(scores)),'prompt_agreement_before_inverse':initial_consistency.tolist(),
                'prompt_agreement_after_inverse':restored_consistency.tolist()})
        masks=np.stack(all_masks);scores=np.stack(all_scores);rows=np.arange(10)
        response=masks[rows,scores.argmax(1)].sum(0)
        mu=float(response[response>0].mean()) if (response>0).any() else 0
        old=(response>=mu/4) if mu>0 else np.zeros_like(response,dtype=bool)
        saved=np.asarray(Image.open(root/'Seal_data/train/SAM_1fg5bg'/(path.stem+'.png')))>128
        gt=np.asarray(Image.open(root/'Seal_data/train/GT'/(path.stem+'.png')))>128
        record={'name':path.stem,'index':int(index),'mu_count':mu,'threshold_count':mu/4,
            'pseudo_area':float(old.mean()),'saved_disagreement':float(np.mean(old!=saved)),
            'prompt_agreement_final':(sample_masks(old[None],original_xy)[0]==labels).tolist(),
            'gt_area_diagnostic':float(gt.mean()),'pseudo_iou_diagnostic':float(np.logical_and(old,gt).sum()/max(np.logical_or(old,gt).sum(),1)),
            'views':views}
        reports.append(record)
        np.savez_compressed(out/f'sam_candidates_{path.stem}.npz',masks=masks,scores=scores,
            direct_masks=np.stack(fallback_masks),direct_scores=np.stack(fallback_scores),xy=original_xy,labels=labels,
            prompt_before=np.stack(before),prompt_after=np.stack(after))
        (out/'sam_audit.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
        print(f'SAM diagnostic {len(reports)}/16: {path.stem} area={old.mean():.5f}',flush=True)
