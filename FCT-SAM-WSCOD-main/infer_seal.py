"""Run the FCT-SAM test pipeline on the Seal RGB test split."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dataset import TestDataset
from model.TestCamoFormer import CamoFormer
from model.decoder.decoder_baseline import Decoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Seal_data/test directory")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    dataset = TestDataset(str(args.data), size=checkpoint.get("input_size", 384), image_dir="RGB")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    model = CamoFormer(cfg=None)
    model.decoder = Decoder(64)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()

    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]
            image = sample["img"].unsqueeze(0).to(device)
            _, _, _, _, output = model(image)
            output = F.interpolate(
                output, size=sample["shape"], mode="bilinear", align_corners=False
            )
            prediction = output.sigmoid().cpu().numpy().squeeze()
            prediction = (prediction - prediction.min()) / (
                prediction.max() - prediction.min() + 1e-8
            )
            name = Path(sample["name"]).name
            Image.fromarray((prediction * 255).astype(np.uint8)).save(args.output / name)
            if (index + 1) % 100 == 0 or index + 1 == len(dataset):
                print(f"{index + 1}/{len(dataset)}", flush=True)


if __name__ == "__main__":
    main()
