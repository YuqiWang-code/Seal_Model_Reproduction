"""Convert Seal's 16-bit thermal TIFFs to the 8-bit RGB input used by NESS-Net."""

import argparse
from pathlib import Path

from infer_seal import index_images, load_thermal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    thermal = index_images(args.seal_dir / "test" / "T")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for count, (stem, source) in enumerate(sorted(thermal.items()), 1):
        load_thermal(source).save(args.output_dir / f"{stem}.png")
        if count % 100 == 0 or count == len(thermal):
            print(f"thermal: {count}/{len(thermal)}", flush=True)


if __name__ == "__main__":
    main()
