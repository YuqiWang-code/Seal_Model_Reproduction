import argparse
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_id")
    parser.add_argument("size", type=int)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    url = f"https://drive.usercontent.google.com/download?id={args.file_id}&export=download&confirm=t"
    curl = shutil.which("curl.exe") or shutil.which("curl")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    chunk_size = (args.size + args.workers - 1) // args.workers

    def download(index):
        start = index * chunk_size
        end = min(args.size - 1, start + chunk_size - 1)
        part = args.output.with_name(args.output.name + f".part{index:02d}")
        subprocess.run([curl, "-sS", "-L", "--fail", "--retry", "3",
                        "--range", f"{start}-{end}", "--output", str(part), url], check=True)
        if part.stat().st_size != end - start + 1:
            raise RuntimeError(f"Incomplete range {index}: {part.stat().st_size} bytes")
        print(f"range {index + 1}/{args.workers} complete", flush=True)
        return part

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        parts = list(pool.map(download, range(args.workers)))
    with args.output.open("wb") as merged:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, merged)
            part.unlink()
    print(f"saved {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
