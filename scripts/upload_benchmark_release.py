#!/usr/bin/env python3
"""upload_benchmark_release.py: push the gated `benchmark/` folder (manifests, frozen config, scores) and the updated card
of the DriveDNA-GPS staging tree to the Hub.

  readme    validate and upload README.md alone
  upload    upload_folder of benchmark/ (42 files, about 330 MB; large parquet through xet)
  upload_extra  upload_folder of models/ (the trained weights, identical to the GitHub export) and code/ (snapshot of GPS--CAN
            plus COMMIT.txt), one commit each; the staged folders are prepared by stage_extra (rsync from the GitHub checkout)
  stage_extra   rsync models/learn and the code snapshot from the GitHub checkout into the staging tree and write code/COMMIT.txt
  check     list benchmark/, models/ and code/ on the Hub, compare the file lists with the staged copies, verify every models/
            and code/ file by sha256 (LFS) or git blob id (regular files) and three random benchmark/ files by download; print gating
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
GPS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GITHUB = os.environ.get("RELEASE_DIR", os.path.join(os.path.dirname(os.path.dirname(GPS_DIR)), "GPS--CAN"))   # the committed code release
EXTRA = {"models": "models", "code": "code"}      # staged folder -> path_in_repo


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def cmd_readme(api, args):
    DatasetCard.load(os.path.join(STAGING, "README.md")).validate(repo_type="dataset")
    r = api.upload_file(path_or_fileobj=os.path.join(STAGING, "README.md"), path_in_repo="README.md", repo_id=REPO, repo_type="dataset",
                        commit_message="Dataset card: trained models (models/learn) and code snapshot (code/)")
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


def cmd_stage_extra(api, args):
    """models/learn and the code snapshot come from the committed GitHub checkout, so the three copies are byte-identical."""
    import subprocess
    head = subprocess.run(["git", "-C", GITHUB, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", GITHUB, "status", "--porcelain"], check=True, capture_output=True, text=True).stdout.strip()
    if dirty:
        sys.exit(f"{GITHUB} has uncommitted changes; commit (and push) first:\n{dirty}")
    os.makedirs(os.path.join(STAGING, "models"), exist_ok=True)
    subprocess.run(["rsync", "-a", "--delete", os.path.join(GITHUB, "models", "learn") + "/", os.path.join(STAGING, "models", "learn") + "/"], check=True)
    subprocess.run(["rsync", "-a", "--delete", "--exclude", ".git", "--exclude", "__pycache__", "--exclude", ".pytest_cache",
                    GITHUB + "/", os.path.join(STAGING, "code") + "/"], check=True)
    with open(os.path.join(STAGING, "code", "COMMIT.txt"), "w") as f:
        f.write(f"https://github.com/WangYuHang-cmd/GPS--CAN at commit {head}\n")
    for name in EXTRA:
        d = os.path.join(STAGING, name)
        print(f"{name}/: {sum(len(fs) for _, _, fs in os.walk(d))} files, {sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs) / 1e6:.1f} MB")
    print("commit", head)


def cmd_upload_extra(api, args):
    for name, path_in_repo in EXTRA.items():
        folder = os.path.join(STAGING, name)
        if not os.path.isdir(folder):
            sys.exit(f"{folder} missing; run stage_extra first")
        msg = {"models": "Trained models: learned motion and signal-set weights (models/learn, as in the code repository)",
               "code": "Code snapshot of the GPS--CAN repository (COMMIT.txt names the commit)"}[name]
        t0 = time.time()
        r = api.upload_folder(folder_path=folder, path_in_repo=path_in_repo, repo_id=REPO, repo_type="dataset", commit_message=msg,
                              delete_patterns=[f"{path_in_repo}/**"])
        print(f"{name}/: upload_folder finished in {time.time() - t0:.0f}s: {r}", flush=True)


def git_blob_id(path):
    data = open(path, "rb").read()
    h = hashlib.sha1()
    h.update(b"blob %d\0" % len(data))
    h.update(data)
    return h.hexdigest()


def cmd_check(api, args):
    files = api.list_repo_files(REPO, repo_type="dataset")
    for top in ("benchmark", "models", "code"):
        hub = sorted(f for f in files if f.startswith(top + "/"))
        root = os.path.join(STAGING, top)
        local = sorted(os.path.relpath(os.path.join(r, f), STAGING) for r, _, fs in os.walk(root) for f in fs) if os.path.isdir(root) else []
        missing = sorted(set(local) - set(hub)); extra = sorted(set(hub) - set(local))
        print(f"{top}/: on hub {len(hub)}, staged {len(local)}, missing {len(missing)} {missing[:3]}, extra {len(extra)} {extra[:3]}")
        if top in ("models", "code") and hub:
            ok = bad = 0
            for i in api.get_paths_info(REPO, hub, repo_type="dataset"):
                p = os.path.join(STAGING, i.path)
                if not os.path.exists(p):
                    continue
                match = (i.lfs.sha256 == sha256(p)) if i.lfs else (i.blob_id == git_blob_id(p))
                ok += match; bad += (not match)
                if not match:
                    print("  MISMATCH", i.path)
            print(f"  {top}/ content check: {ok} match, {bad} mismatch")
    print(f"hub files {len(files)}")
    local = sorted(os.path.relpath(os.path.join(r, f), STAGING) for r, _, fs in os.walk(FOLDER) for f in fs)
    random.seed(int(time.time()))
    for rel in random.sample(local, 3):
        p = hf_hub_download(REPO, rel, repo_type="dataset", cache_dir=os.path.join(STAGING, ".cache"))
        print(rel, "sha256 match" if sha256(p) == sha256(os.path.join(STAGING, rel)) else "MISMATCH")
    info = api.dataset_info(REPO)
    print("gated:", info.gated, "private:", info.private, "sha:", info.sha[:8])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["readme", "upload", "upload_large", "stage_extra", "upload_extra", "check"])
    args = ap.parse_args()
    api = HfApi()
    {"readme": cmd_readme, "upload": cmd_upload, "upload_large": cmd_upload_large, "stage_extra": cmd_stage_extra,
     "upload_extra": cmd_upload_extra, "check": cmd_check}[args.cmd](api, args)


if __name__ == "__main__":
    sys.exit(main())
