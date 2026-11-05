#!/usr/bin/env python3
"""upload_benchmark_release.py: push the gated `benchmark/` folder (manifests, frozen config, scores) and the updated card
of the DriveDNA-GPS staging tree to the Hub.

  readme    validate and upload README.md alone
  upload    upload_folder of benchmark/ (42 files, about 330 MB; large parquet through xet)
  check     list benchmark files on the Hub and compare three sha256 against the staged copies; print gating
Run with the miniconda python (huggingface_hub 0.36.2); HF_XET_CACHE on /data/backup (root disk is nearly full).
"""
import argparse
import hashlib
import os
import random
import sys
import time

os.environ.setdefault("HF_XET_CACHE", "/data/backup/hf_xet_cache")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_UPDATE_CHECK", "1")

from huggingface_hub import DatasetCard, HfApi, hf_hub_download  # noqa: E402

REPO = "HenryYHW/CANTraj"   # renamed from HenryYHW/DriveDNA-GPS on 2026-11-05 (the Hub redirects the old name)
STAGING = "/data/backup/DriveDNA-GPS_hf"
FOLDER = os.path.join(STAGING, "benchmark")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def cmd_readme(api, args):
    DatasetCard.load(os.path.join(STAGING, "README.md")).validate(repo_type="dataset")
    r = api.upload_file(path_or_fileobj=os.path.join(STAGING, "README.md"), path_in_repo="README.md", repo_id=REPO, repo_type="dataset",
                        commit_message="Dataset card: CANTraj-Bench version 1.1 section, agent study and reports")
    print("README uploaded:", r)


def cmd_upload(api, args):
    t0 = time.time()
    r = api.upload_folder(folder_path=FOLDER, path_in_repo="benchmark", repo_id=REPO, repo_type="dataset",
                          commit_message="CAN-to-GPS benchmark v1: manifests, frozen configuration and per-episode scores")
    print(f"upload_folder finished in {time.time() - t0:.0f}s: {r}", flush=True)


def cmd_upload_large(api, args):
    """Resumable upload of benchmark/ through a hard-linked root (upload_large_folder has no path_in_repo)."""
    import shutil, subprocess
    root = "/data/backup/DriveDNA-GPS_up"
    if os.path.isdir(os.path.join(root, "benchmark")):
        shutil.rmtree(os.path.join(root, "benchmark"))
    os.makedirs(root, exist_ok=True)
    subprocess.run(["cp", "-al", FOLDER, os.path.join(root, "benchmark")], check=True)
    t0 = time.time()
    api.upload_large_folder(repo_id=REPO, repo_type="dataset", folder_path=root, num_workers=4, print_report_every=60)
    print(f"upload_large_folder finished in {time.time() - t0:.0f}s", flush=True)


def cmd_check(api, args):
    files = api.list_repo_files(REPO, repo_type="dataset")
    hub = sorted(f for f in files if f.startswith("benchmark/"))
    local = sorted(os.path.relpath(os.path.join(r, f), STAGING) for r, _, fs in os.walk(FOLDER) for f in fs)
    missing = sorted(set(local) - set(hub)); extra = sorted(set(hub) - set(local))
    print(f"hub files {len(files)}, benchmark on hub {len(hub)}, local {len(local)}, missing {len(missing)} {missing[:3]}, extra {len(extra)} {extra[:3]}")
    random.seed(int(time.time()))
    for rel in random.sample(local, 3):
        p = hf_hub_download(REPO, rel, repo_type="dataset", cache_dir=os.path.join(STAGING, ".cache"))
        print(rel, "sha256 match" if sha256(p) == sha256(os.path.join(STAGING, rel)) else "MISMATCH")
    info = api.dataset_info(REPO)
    print("gated:", info.gated, "private:", info.private)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["readme", "upload", "upload_large", "check"])
    args = ap.parse_args()
    api = HfApi()
    {"readme": cmd_readme, "upload": cmd_upload, "upload_large": cmd_upload_large, "check": cmd_check}[args.cmd](api, args)


if __name__ == "__main__":
    sys.exit(main())
