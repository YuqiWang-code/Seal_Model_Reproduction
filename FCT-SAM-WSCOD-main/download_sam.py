"""Download the official SAM ViT-H checkpoint using HTTP byte ranges."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import subprocess


def main():
    target = Path("pretrained/sam_vit_h_4b8939.pth")
    total = 2564550879
    if target.exists() and target.stat().st_size == total:
        return
    parts = target.parent / "sam_parts"
    parts.mkdir(parents=True, exist_ok=True)
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

    def fetch(index):
        start, stop = total * index // 32, total * (index + 1) // 32 - 1
        path = parts / f"{index:02d}"
        for attempt in range(5):
            received = path.stat().st_size if path.exists() else 0
            if received == stop - start + 1:
                break
            with path.open("ab") as output:
                result = subprocess.run(["curl", "--http1.1", "-fsSL", "--max-time", "900",
                                         "--range", f"{start + received}-{stop}", url], stdout=output)
            if result.returncode == 0:
                break
        if path.stat().st_size != stop - start + 1:
            raise RuntimeError(f"Incomplete SAM download range {index}")
        print(f"SAM download: part {index + 1}/32 ready", flush=True)

    with ThreadPoolExecutor(max_workers=32) as pool:
        list(pool.map(fetch, range(32)))
    with target.with_suffix(".tmp").open("wb") as output:
        for index in range(32):
            with (parts / f"{index:02d}").open("rb") as source:
                shutil.copyfileobj(source, output)
    target.with_suffix(".tmp").replace(target)
    for path in parts.iterdir():
        path.unlink()
    parts.rmdir()


if __name__ == "__main__":
    main()
