import argparse
import os


parser = argparse.ArgumentParser(description="LRDNet on Seal")
parser.add_argument("--stage", type=int, choices=[1, 2], default=1)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--data_root", type=str,
                    default="/share_datasets/Seal_Model_Reproduction/Seal_data")
parser.add_argument("--experiment_root", type=str, default="experiments/SCOD")
parser.add_argument("--soft_gt_root", type=str, default=None)
parser.add_argument("--train_image_root", type=str, default=None)
parser.add_argument("--train_label_root", type=str, default=None)
parser.add_argument("--val_root", type=str, default=None)
parser.add_argument("--test_root", type=str, default=None)
parser.add_argument("--test_image_dir", type=str, default="RGB")
parser.add_argument("--validate_epoch", type=int, default=0)
args = parser.parse_args()

param = {
    "device": 0,
    "second_time": args.stage == 2,
    "epoch": args.epochs,
    "batch_size": args.batch_size,
    "lr_config": {"beta": [0.5, 0.999], "lr": 1e-4, "lr_dis": 1e-5,
                  "decay_rate": 0.75, "decay_epoch": 20, "gamma": 0.98},
    "save_epoch": 10,
    "validate_epoch": args.validate_epoch,
    "trainsize": 384,
    "neck_channel": 32,
    "num_workers": 4,
    "backbone": "R50",
    "decoder": "base_decoder",
    "seed": 1024,
    "pretrain": "pth/resnet50-0676ba61.pth",
}

param["log_path"] = os.path.join(args.experiment_root, f"stage{args.stage}")
param["ckpt_save_path"] = os.path.join(param["log_path"], "models")
param["soft_gt_root"] = args.soft_gt_root or os.path.join(args.experiment_root, "soft_gt")
param["paths"] = {
    "image_root": args.train_image_root or "data_original/TrainDataset/Imgs",
    "gt_root": param["soft_gt_root"] if param["second_time"] else
               (args.train_label_root or "data_original/Scribble"),
    "val_root": args.val_root or os.path.join(args.data_root, "val"),
    "test_root": args.test_root or os.path.join(args.data_root, "test"),
}
param["checkpoint"] = args.checkpoint
param["eval_save_path"] = os.path.join(args.experiment_root, "test_maps")
param["test_image_dir"] = args.test_image_dir
