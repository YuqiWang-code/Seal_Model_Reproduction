import os
import warnings

import torch
from sklearn.metrics import auc
from tqdm import tqdm

from utils.metrics import (
    PD_FA,
    ROCMetric,
    SamplewiseSigmoidMetric,
    SegmentationMetricTPFNFP,
    mIoUMetric,
)
from utils.tools import para_parser, test_head
warnings.filterwarnings("ignore")


args = para_parser()
# ------------------------------------------------------------------
save_root_path = r'result/xx'
args.dataset = save_root_path.split('/')[-1].split('_')[0]
args.paper = 'DURD'
pretrained_model_dir = os.path.join(
    save_root_path, 'Saved_parameters.pt').replace("\\", "/")
model, test_dataset, test_dataloader = test_head(args, pretrained_model_dir)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
# ------------------------------------------------------------------
# metrics
mIoU_metric = mIoUMetric(1)
nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)
metrics = SegmentationMetricTPFNFP(nclass=1)
metric_roc = ROCMetric(nclass=1, bins=200)
PD_FA = PD_FA(1, 200)


def testing():
    model.eval()
    with torch.no_grad():
        mIoU_metric.reset()
        nIoU_metric.reset()
        metrics.reset()
        metric_roc.reset()
        PD_FA.reset()
        for name, batch_image, batch_label in tqdm(
                test_dataloader):

            batch_image = batch_image.to(device)
            batch_label = batch_label.to(device)

            outputs = model(batch_image)
            pred = outputs[-1]

            mIoU_metric.update(pred, batch_label)
            nIoU_metric.update(preds=pred, labels=batch_label)
            metrics.update(labels=batch_label, preds=pred)
            metric_roc.update(labels=batch_label, preds=pred)
            PD_FA.update(preds=pred, labels=batch_label)

        _, miou = mIoU_metric.get()
        _, nIoU = nIoU_metric.get()
        _, prec, recall, fmeasure = metrics.get()
        tpr, fpr = metric_roc.get()
        auc_value = auc(fpr, tpr)
        FA_set, PD_set = PD_FA.get(len(test_dataset))
        FA = FA_set[0] * 1000000
        PD = PD_set[0]
        print(
            'Precision: %.4f | Recall: %.4f | F-measure: %.4f | mIoU: %.4f | nIoU: %.4f | AUC: %.4f | P_d: %.4f | F_a: %.4f'
            % (prec, recall, fmeasure, miou, nIoU, auc_value, PD, FA))


if __name__ == "__main__":
    testing()
