import json
import py_sod_metrics
from tqdm import tqdm
from pprint import pprint
from config import param as option
from dataset.dataloader import EvalDataset

# PF,NF

def eval_function(eval_loader):
    FM = py_sod_metrics.Fmeasure()
    WFM = py_sod_metrics.WeightedFmeasure()
    SM = py_sod_metrics.Smeasure()
    EM = py_sod_metrics.Emeasure()
    MAE = py_sod_metrics.MAE()

    for pred, gt in tqdm(eval_loader):
        FM.step(pred=pred, gt=gt)
        WFM.step(pred=pred, gt=gt)
        SM.step(pred=pred, gt=gt)
        EM.step(pred=pred, gt=gt)
        MAE.step(pred=pred, gt=gt)

    fm = FM.get_results()["fm"]
    pr = FM.get_results()["pr"]
    wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]
    em = EM.get_results()["em"]
    mae = MAE.get_results()["mae"]

    results = {
        "Smeasure": sm,
        "wFmeasure": wfm,
        "MAE": mae,
        "adpEm": em["adp"],
        "meanEm": em["curve"].mean(),
        "maxEm": em["curve"].max(),
        "adpFm": fm["adp"],
        "meanFm": fm["curve"].mean(),
        "maxFm": fm["curve"].max(),
        "precision": pr["p"].mean(),
        "recall": pr["r"].mean()
    }
    return results


if __name__ == '__main__':
    test_epoch_num = option['checkpoint'].split('/')[-1].split('_')[0]
    for dataset in option['datasets']:
        gt_root = option['paths']['test_root'] + dataset + '/GT_Object/'
        eval_loader = EvalDataset(option['eval_save_path'] + f'{test_epoch_num}_epoch/' + dataset + '/', gt_root)
        res = eval_function(eval_loader)
        pprint(res)
        with open(option['eval_save_path'] + f'{test_epoch_num}_epoch/' + f'/results_{dataset}_epoch.json', 'w') as f:
            f.write(json.dumps(res))
