import ast
import inspect
import json
import sys
import textwrap
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

root = Path('/share_datasets/Seal_Model_Reproduction')
code = root / 'FCT-SAM-WSCOD-main'
sys.path.insert(0, str(code))
from dataset import TestDataset
from model.TestCamoFormer import CamoFormer
import model.decoder.decoder_baseline as decoder
import utils.metrics as measure

torch.set_num_threads(2)
outdir = root / 'results/FCT-SAM-WSCOD/diagnosis_20261103'
outdir.mkdir(parents=True, exist_ok=True)
checkpoint = code / 'pretrain/Pvtv2_TE_new_dataset_batch4_epoch60.pth'
model = CamoFormer(None)
model.decoder = decoder.Decoder(64)
model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['state_dict'], strict=True)
model.cuda().eval()
original_text = textwrap.dedent(inspect.getsource(decoder.CrossStrengthen.forward)).replace(
    'value_correlation = vx @ vx.transpose(-2, -1)\n    out = ((attnx @ attny @ value_correlation) @ vy) * self.temperature3',
    'out = (attnx @ attny @ vx) @ (vx.transpose(-2, -1) @ vy) * self.temperature3')
namespace = dict(vars(decoder))
exec(original_text, namespace)
original_forward = namespace['forward']

def use_original(enabled):
    for module in model.modules():
        if isinstance(module, decoder.CrossStrengthen):
            module.forward = types.MethodType(original_forward if enabled else decoder.CrossStrengthen.forward, module)

@torch.inference_mode()
def prediction(sample):
    output = model(sample['img'][None].cuda())[-1]
    output = F.interpolate(output, sample['shape'], mode='bilinear', align_corners=False)
    p = output.sigmoid().cpu().numpy().squeeze()
    return (p-p.min())/(p.max()-p.min()+1e-8)

seal = TestDataset(str(root/'Seal_data/test'), 384, image_dir='RGB')
seal.list.sort()
comparison=[]
with torch.inference_mode():
    for index in [0, len(seal)//2, len(seal)-1]:
        sample=seal[index]
        use_original(False); a=prediction(sample)
        use_original(True); b=prediction(sample)
        saved=np.asarray(Image.open(root/'results/FCT-SAM-WSCOD/Seal/test_maps'/Path(sample['name']).name))
        png=(b*255).astype(np.uint8)
        comparison.append({'name':Path(sample['name']).name,'reassociated_max_abs':float(np.max(np.abs(a-b))),
            'saved_vs_author_png_mae_uint8':float(np.abs(png.astype(float)-saved).mean()),
            'saved_vs_author_png_different_fraction':float(np.mean(png!=saved))})

source = root/'LRDNet-master/data_original/TestDataset/CAMO'
data=TestDataset(str(source),384,image_dir='Imgs')
pred_dir=outdir/'CAMO_author_predictions';pred_dir.mkdir(exist_ok=True)
fm,sm,em,mae=measure.Fmeasure(),measure.Smeasure(),measure.Emeasure(),measure.MAE()
with torch.inference_mode():
    for index in range(len(data)):
        sample=data[index];p=prediction(sample);gt=sample['label'].numpy().squeeze()
        Image.fromarray((p*255).astype(np.uint8)).save(pred_dir/Path(sample['name']).name)
        for metric in [fm,sm,em,mae]: metric.step(pred=p*255,gt=gt*255)
        if (index+1)%50==0:print(f'CAMO {index+1}/{len(data)}',flush=True)
report={'checkpoint':str(checkpoint),'source':'CAMO test (existing author test dataset)', 'source_count':len(data),
    'author_metrics':{'S':float(sm.get_results()['sm']),'adpF':float(fm.get_results()['fm']['adp']),
        'meanE':float(em.get_results()['em']['curve'].mean()),'MAE':float(mae.get_results()['mae'])},
    'comparison_to_saved_seal':comparison,'source_metrics_role':'source-domain diagnostic only; author metric definitions, not Seal summary table',
    'source_prediction_dir':str(pred_dir)}
(outdir/'source_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2),flush=True)
