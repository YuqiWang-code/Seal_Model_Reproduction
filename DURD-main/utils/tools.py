import os
import argparse
import torch
import torch.nn as nn
from torch.nn import init
from torch.utils.data import DataLoader
from datetime import datetime

from utils.dataloader import IR_Dataset
from utils.DURD import DURD


def para_parser():
    arg_parser = argparse.ArgumentParser(description='small target detection')
    arg_parser.add_argument('--epochs',
                            type=int,
                            default=300,
                            metavar='N',
                            help='number of epochs to train (default: 300)')
    arg_parser.add_argument('--batch_size', default=8)
    arg_parser.add_argument('--test_batch_size',
                            type=int,
                            default=8,
                            metavar='N',
                            help='input batch size for testing (default: 32)')
    arg_parser.add_argument('--in_channels',
                            type=int,
                            default=1,
                            help='in_channel=1 for pre-process')
    arg_parser.add_argument('--dataset_root_path',
                            type=str,
                            default='dataset',
                            help='dataset_root_path')
    arg_parser.add_argument('--lr',
                            type=float,
                            default=0.001,
                            metavar='LR',
                            help='learning rate (default: 0.001)')
    arg_parser.add_argument('--min_lr',
                            default=1e-5,
                            type=float,
                            help='minimum learning rate')
    args = arg_parser.parse_args()

    return args


def train_head(args):
    train_dataset = IR_Dataset(args.dataset_root_path,
                               args.dataset,
                               train=True,
                               image_channel=args.in_channels,
                               splite_result_for_model=args.paper)
    test_dataset = IR_Dataset(args.dataset_root_path,
                              args.dataset,
                              test=True,
                              image_channel=args.in_channels,
                              splite_result_for_model=args.paper)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False)
    # ------------------------------------------------------------------
    model = DURD(in_channel=args.in_channels)

    model.apply(weights_init_xavier)
    # ------------------------------------------------------------------

    return model, train_dataloader, test_dataloader


def test_head(args, model_dir):
    test_dataset = IR_Dataset(args.dataset_root_path,
                              args.dataset,
                              test=True,
                              image_channel=args.in_channels,
                              splite_result_for_model=args.paper)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.test_batch_size,
                                 shuffle=False)
    # ------------------------------------------------------------------
    model = DURD(in_channel=args.in_channels)

    model.load_state_dict(torch.load(
        model_dir, map_location="cpu"), strict=True)

    return model, test_dataset, test_dataloader


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.xavier_normal(m.weight.data)


def SoftIoULoss(pred, target):
    pred = torch.sigmoid(pred)
    smooth = 1
    intersection = pred * target
    loss = (intersection.sum() + smooth) / (pred.sum() + target.sum() -
                                            intersection.sum() + smooth)
    loss = 1 - loss.mean()
    return loss


class DURD_Loss(nn.Module):

    def __init__(self,
                 pred_loss_fn=SoftIoULoss,
                 recon_loss_fn=None,
                 recon_weight=0.1):
        super().__init__()
        self.pred_loss_fn = pred_loss_fn
        self.recon_loss_fn = recon_loss_fn if recon_loss_fn is not None else nn.MSELoss()
        self.recon_weight = float(recon_weight)

    def forward(self, model_outputs, image, label):

        if not isinstance(model_outputs, (tuple, list)):
            raise ValueError("model_outputs must be a tuple/list")

        _, _, x_recon, s_prior, multi_preds, f_dec, final_pred = model_outputs

        loss_mse = self.recon_loss_fn(x_recon, image)
        loss_sparse = self.pred_loss_fn(s_prior, label)

        loss_pred = 0.0
        if multi_preds is not None:
            if isinstance(multi_preds, (tuple, list)) and len(multi_preds) > 0:
                inv_n = 1.0 / float(len(multi_preds))
                for pred in multi_preds:
                    loss_pred = loss_pred + inv_n * \
                        self.pred_loss_fn(pred, label)
            else:
                # keep backward-compatible behavior if M is an empty list / None
                loss_pred = loss_pred + 0.0

        loss_pred = loss_pred + self.pred_loss_fn(f_dec, label)
        loss_dcf = self.pred_loss_fn(final_pred, label)

        loss_total = self.recon_weight * loss_mse + loss_sparse + loss_pred + loss_dcf

        return loss_total


def make_dir(dataset):
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")

    save_path = "%s_%s" % (dataset, dt_string)
    os.makedirs('result/%s' % (save_path), exist_ok=True)
    return save_path
