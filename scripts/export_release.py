"""Export the benchmark code to a clean release tree and run the identifier gate.

  python scripts/export_release.py --out /path/to/GPS--CAN [--check-only]

Copies: src/cantraj, scripts, configs, specs, tests, pytest.ini, README.md, LICENSE, requirements.txt, ENVIRONMENT.md,
reports/full_v1_summary.md, reports/failure_analysis_v1.md, reports/paper/ (card, methods, tables, figures and the
figure data). Never copies: manifests/ (published through the gated data repository), _private, predictions, models,
logs, progress.md, caches, __pycache__. Gate: every text file is scanned for raw device / route identifiers, local
absolute paths and the Controlled-supplement city; any hit aborts the export.
"""
from __future__ import annotations

import argparse
import codecs
import os
import re
import shutil
import sys

GPS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CITY = codecs.decode("Gnzcn", "rot13")   # the Controlled supplement's city; kept out of every released text, this one included
INCLUDE = ["src/cantraj", "scripts", "configs", "specs", "tests", "pytest.ini", "README.md", "LICENSE", "requirements.txt",
           "ENVIRONMENT.md", "reports/full_v1_summary.md", "reports/failure_analysis_v1.md", "reports/paper",
           # ICDE study: method reproduction notes, input provenance and aggregate tables only (no per-episode or
           # per-tile files: the tile list shows the driven regions and stays with the gated data)
           "reports/icde/methods_repro.md", "reports/icde/input_provenance.md", "reports/icde/tables/main_board.md",
           "reports/icde/tables/main_board.csv", "reports/icde/tables/methods.csv",
           "reports/icde/tables/sig_signal_use.csv", "reports/icde/ep_tie_sensitivity.csv", "reports/icde/ep_val_selection.csv",
           "reports/icde/opd2_author_samples.csv", "reports/icde/macrotracking_author_sample.csv",
           "reports/icde/tables/pairs.csv", "reports/icde/tables/topk_w05.csv", "reports/icde/tables/signal_sets_w05.csv",
           "reports/icde/tables/signal_gains_w05.csv", "reports/icde/tables/signal_reference_w05.csv",
           "reports/icde/tables/cohort_horizon.csv", "reports/icde/tables/cohort_pairs.csv", "reports/icde/tables/cost.csv",
           "reports/icde/tables/cost_accuracy.csv", "reports/icde/tables/map_prep_cost.csv", "reports/icde/tables/robustness_w05.csv",
           "reports/icde/tables/a3_visibility.csv", "reports/icde/tables/spy_dewri_diag.csv", "reports/icde/spy_val_selection.csv",
           # agent study (supplementary cohort): protocol, results report and aggregate tables; logs and caches stay in the data package
           "reports/icde/agent_protocol_v1.md", "reports/icde/agent_results.md", "reports/icde/tables/agent_w05.csv",
           "reports/icde/tables/agent_pairs_w05.csv", "reports/icde/tables/agent_status_w05.csv", "reports/icde/tables/agent_cohort_counts.csv",
           "reports/icde/tables/agent_cohort_counts_val.csv", "reports/icde/tables/agent_selector_val.csv", "reports/icde/tables/agent_identity_check.csv",
           "reports/icde/tables/agent_live_check.csv", "reports/icde/tables/agent_pilot_summary.csv"]
EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", "_private", ".git"}
# machine-specific helpers and the Hugging Face staging / upload scripts are not part of the code release
EXCLUDE_FILES = {"progress.md", "memwatch.sh", "stage_events_release.py", "upload_events_release.py", "patch_card_events.py",
                 "tile_pilot_maps.py", "make_pilot_report.py", "run_pilot.sh", "stop_chain2_before_full.sh",
                 "stage_public_package.py", "upload_public_package.py"}       # anonymous-package tooling: never in this (named) repository
TEXT_EXT = {".py", ".md", ".txt", ".yaml", ".yml", ".csv", ".ini", ".cfg", ".toml", ".json", ".sh"}
DATA_EXT = {".csv", ".json", ".md", ".txt"}
OPENPILOT = "$OPENPILOT_DIR"
GATE = [
    ("dongle id (16 hex)", re.compile(r"(?<![0-9a-zA-Z.])(?=[0-9]*[a-f])[0-9a-f]{16}(?![0-9a-zA-Z])")),   # hex with a letter: not a float
    ("route timestamp", re.compile(r"\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2}")),
    ("local path", re.compile(r"/home/[a-z]+/|/data/datasets/raw|hf_staging_video|_local/videos")),
    ("controlled city", re.compile(re.escape(CITY))),              # any occurrence, also inside a regex literal
    ("coordinate column", re.compile(r"\blat0\b|\blon0\b|\blatitude\b.*,.*\blongitude\b")),   # data and documents only
]
GATE_ALLOW = {  # files where a pattern is expected and harmless (sha256 prefixes in the spec look like hex ids)
    "specs/benchmark_v1.md": {"dongle id (16 hex)"},
    "configs/benchmark_v1.yaml": {"dongle id (16 hex)"},
    "tests/test_splits.py": {"controlled city", "local path"},  # the test that asserts their absence
    "reports/paper/benchmark_card_draft.md": {"coordinate column"},  # names the forbidden signals (latitude, longitude)
    "reports/paper/methods_draft.md": {"coordinate column"},
    "tests/test_manifest_v1.py": {"dongle id (16 hex)"},
    "configs/agent/protocol_v1.md": {"dongle id (16 hex)"},        # the tool-runner code sha quoted in the frozen protocol
    "reports/icde/agent_protocol_v1.md": {"dongle id (16 hex)"},
    "scripts/export_release.py": {"dongle id (16 hex)", "route timestamp", "local path", "controlled city", "coordinate column"},
}


def transform(rel, txt):
    """Release-time edits of text files: the authors' interpreter becomes an environment variable, the Controlled
    supplement's city is not named, hard-coded cache roots go through cantraj.paths."""
    ext = os.path.splitext(rel)[1].lower()
    txt = txt.replace(f"PYTHONPATH={OPENPILOT} ", "").replace(f"PYTHONPATH={OPENPILOT}:", "PYTHONPATH=")
    txt = txt.replace(f"{OPENPILOT}/.venv/bin/python", "${PYTHON:-python3}" if ext == ".sh" else "python")
    txt = txt.replace(OPENPILOT, "$OPENPILOT_DIR")
    txt = txt.replace(CITY, "the Controlled-supplement city")
    if ext == ".sh":
        txt = txt.replace("/data/backup/DriveDNA-GPS_bench", "${CANTRAJ_BENCH:-/data/backup/DriveDNA-GPS_bench}")
        txt = txt.replace("/home/henry/Desktop/Drive/DriveDNA/experiments/logs", "${CANTRAJ_LOGS:-logs}")
        txt = txt.replace("cd /home/henry/Desktop/Drive/DriveDNA/GPS", 'cd "$(dirname "$0")/.."')
        txt = txt.replace("--out /home/henry/Desktop/Drive/GPS--CAN", "--out ${RELEASE_DIR:-../GPS--CAN}")
    if rel == "scripts/m0_vehicle_params.py":
        txt = txt.replace('BENCH = "/data/backup/DriveDNA-GPS_bench"', 'BENCH = os.environ.get("CANTRAJ_BENCH", "/data/backup/DriveDNA-GPS_bench")')
        txt = txt.replace('DRIVES = "/home/henry/Desktop/Drive/DriveDNA/hf_staging/index/drives.parquet"',
                          'DRIVES = os.path.join(os.environ.get("CANTRAJ_DRIVEDNA", "."), "index", "drives.parquet")')
    if rel == "scripts/m0_redecode_rlog.py":
        txt = txt.replace('sys.path.insert(0, "/home/henry/Desktop/Drive/DriveDNA/code/preprocessing")',
                          'sys.path.insert(0, os.environ.get("DRIVEDNA_CODE", "code/preprocessing"))   # DriveDNA preprocessing helpers')
    # manifests referenced by absolute path -> the CANTRAJ_MANIFESTS directory; Natural Earth polygons -> environment variables
    txt = re.sub(r'os.path.join(os.environ.get("CANTRAJ_MANIFESTS", "manifests"), "([^")]+)"',
                 lambda m: f'os.path.join(os.environ.get("CANTRAJ_MANIFESTS", "manifests"), "{m.group(1)}")', txt)
    txt = re.sub(r'"/home/henry/Desktop/Drive/LKA_Detection/lkas_analysis_output/([a-z_]+)\.geojson"',
                 lambda m: f'os.environ.get("CANTRAJ_{m.group(1).upper()}_GEOJSON", "{m.group(1)}.geojson")   # Natural Earth polygons', txt)
    return txt


def iter_files():
    for rel in INCLUDE:
        src = os.path.join(GPS_DIR, rel)
        if os.path.isfile(src):
            yield rel
            continue
        for root, dirs, files in os.walk(src):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for f in files:
                if f in EXCLUDE_FILES or f.endswith((".pyc", ".pyo", ".so")):              # .so: the DTW kernel is built on first use
                    continue
                yield os.path.relpath(os.path.join(root, f), GPS_DIR)


def read_text(rel):
    return transform(rel, open(os.path.join(GPS_DIR, rel), encoding="utf-8", errors="replace").read())


def gate(rel):
    ext = os.path.splitext(rel)[1].lower()
    if ext not in TEXT_EXT:
        return []
    try:
        txt = read_text(rel)
    except OSError:
        return []
    hits = []
    for name, rx in GATE:
        if name in GATE_ALLOW.get(rel, set()):
            continue
        if name == "coordinate column" and ext not in DATA_EXT:
            continue
        m = rx.search(txt)
        if m:
            line = txt.count("\n", 0, m.start()) + 1
            hits.append((name, line, m.group(0)[:60]))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(GPS_DIR)), "GPS--CAN"))
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--clean", action="store_true", help="remove everything in --out except .git before copying")
    args = ap.parse_args()
    files = sorted(set(iter_files()))
    bad = [(rel, h) for rel in files for h in gate(rel)]
    for rel, (name, line, frag) in bad:
        print(f"GATE HIT  {rel}:{line}  {name}: {frag!r}")
    print(f"identifier gate: {len(bad)} hits in {len(files)} files")
    if bad:
        sys.exit(1)
    if args.check_only:
        return
    os.makedirs(args.out, exist_ok=True)
    if args.clean:
        for name in os.listdir(args.out):
            if name in (".git", ".gitignore"):
                continue
            p = os.path.join(args.out, name)
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    n = 0
    for rel in files:
        dst = os.path.join(args.out, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.splitext(rel)[1].lower() in TEXT_EXT:
            with open(dst, "w", encoding="utf-8") as f:
                f.write(read_text(rel))
            shutil.copymode(os.path.join(GPS_DIR, rel), dst)
        else:
            shutil.copy2(os.path.join(GPS_DIR, rel), dst)
        n += 1
    print(f"exported {n} files -> {args.out}")


if __name__ == "__main__":
    main()
