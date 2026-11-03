import json
from pathlib import Path
import cv2
import numpy as np

root=Path('/share_datasets/Seal_Model_Reproduction')
out=root/'results/FCT-SAM-WSCOD/diagnosis_20261103'
records=json.loads((out/'sam_audit.json').read_text())
results=[];selected_before=[];selected_after=[];inverse_changed=[]
for record in records:
    name=record['name'];z=np.load(out/f'sam_candidates_{name}.npz')
    xy=np.rint(z['xy']).astype(int);labels=z['labels'];votes=[];refined=direct=discarded=0
    for view in range(10):
        selected=int(np.argmax(z['scores'][view]))
        selected_before.append(z['prompt_before'][view,selected].tolist())
        selected_after.append(z['prompt_after'][view,selected].tolist())
        inverse_changed.extend((z['prompt_before'][view]!=z['prompt_after'][view]).ravel().tolist())
        chosen=None
        for group,score_key in [('masks','scores'),('direct_masks','direct_scores')]:
            masks=z[group][view]
            agreement=masks[:,xy[:,1],xy[:,0]]==labels
            valid=np.flatnonzero(agreement.all(1))
            if valid.size:
                selected=valid[np.argmax(z[score_key][view,valid])]
                chosen=masks[selected];refined+=group=='masks';direct+=group=='direct_masks'
                break
        if chosen is None:discarded+=1
        else:votes.append(chosen)
    accepted=False;new_area=None;new_iou=None
    if votes:
        response=np.stack(votes).mean(0);positive=response>0
        threshold=float(response[positive].mean()/4)
        filtered=response>=threshold
        accepted=bool(np.array_equal(filtered[xy[:,1],xy[:,0]],labels.astype(bool)))
        new_area=float(filtered.mean())
        gt=cv2.imread(str(root/'Seal_data/train/GT'/f'{name}.png'),0)>128
        new_iou=float(np.logical_and(filtered,gt).sum()/max(np.logical_or(filtered,gt).sum(),1))
        if accepted:cv2.imwrite(str(out/f'consistent_preview_{name}.png'),filtered.astype(np.uint8)*255)
    results.append({'name':name,'old_area':record['pseudo_area'],'old_iou_diagnostic':record['pseudo_iou_diagnostic'],
        'refined_views':refined,'direct_fallback_views':direct,'discarded_views':discarded,'accepted':accepted,
        'new_area':new_area,'new_iou_diagnostic':new_iou})
before=np.asarray(selected_before);after=np.asarray(selected_after)
report={'sample_selection':'16 train image indices fixed by linspace before SAM prediction; GT metrics diagnostic only',
 'n_images':len(results),'n_views':len(before),'selected_fg_consistency_before':float(before[:,0].mean()),
 'selected_bg_false_positive_before':float(1-before[:,1:].mean()),
 'selected_bg_false_positive_after':float(1-after[:,1:].mean()),
 'candidate_prompt_bit_change_due_to_inverse':float(np.mean(inverse_changed)),
 'consistent_output_images':sum(x['accepted'] for x in results),'images':results}
(out/'candidate_analysis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
