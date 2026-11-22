"""Download + prep all of VRGaze, one subject at a time: download the tar (resumable), extract, prep to .npz,
delete the extracted JPGs (the tar stays, so prep can be re-run). Safe to stop and restart.

    python training/prep/fetch_vrgaze.py data/vrgaze data/prepped        -> data/prepped/vrgaze/, data/prepped/vrgaze_val/

Disk: ~77 GB of tars + ~3-4 GB of npz; at most one subject's JPGs (~2.5 GB) on disk at a time.
"""
import json
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import vrgaze  # noqa: E402

BASE = "https://huggingface.co/datasets/gazeshift/VRGaze/resolve/main"
API = "https://huggingface.co/api/datasets/gazeshift/VRGaze/tree/main"


def listing(path):
    with urllib.request.urlopen(f"{API}/{path}", timeout=30) as r:
        return json.load(r)


def fetch(url, dst, size=None):
    if dst.exists() and (size is None or dst.stat().st_size == size):
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        if subprocess.run(["curl.exe", "-sSL", "-C", "-", "--retry", "5", "-o", str(dst), url]).returncode == 0 \
                and (size is None or dst.stat().st_size == size):
            return
        print(f"  retry {attempt + 1} for {dst.name}", flush=True)
        time.sleep(10)
    raise RuntimeError(f"download failed: {url}")


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    t0 = time.time()
    for split, dst in (("val", out / "vrgaze_val"), ("train", out / "vrgaze")):
        fetch(f"{BASE}/metadata_{split}.csv", src / f"metadata_{split}.csv")
        files = sorted(listing(split), key=lambda f: f["path"])
        subjects = {f["path"].split("/")[-1].removesuffix(".tar.gz"): f for f in files if f["path"].endswith(".tar.gz")}
        pairs = vrgaze.read_pairs(src, split, set(subjects))
        done_gb = 0.0
        for i, (subject, f) in enumerate(sorted(subjects.items()), 1):
            npz = dst / f"{subject}.npz"
            if npz.exists():
                continue
            tar = src / f["path"]
            fetch(f"{BASE}/{f['path']}", tar, f["size"])
            done_gb += f["size"] / 1e9
            folder = src / split / subject
            if not folder.exists():
                with tarfile.open(tar) as t:
                    t.extractall(src / split, filter="data")
            vrgaze.prep_subjects(src, dst, split, {subject}, pairs)
            shutil.rmtree(folder, ignore_errors=True)
            print(f"[{split} {i}/{len(subjects)}] {subject} done · {(time.time() - t0) / 60:.0f} min elapsed", flush=True)
    print(f"ALL DONE in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
