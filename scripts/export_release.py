"""Export the benchmark code to a clean release tree and run the identifier gate.

  python scripts/export_release.py --out /path/to/GPS--CAN [--check-only]

Copies: src/cantraj, scripts, configs, specs, tests, pytest.ini, README.md, LICENSE, requirements.txt, ENVIRONMENT.md,
reports/full_v1_summary.md, reports/failure_analysis_v1.md, reports/paper/ (card, methods, tables, figures and the
figure data), and the trained weights `models/learn/` from the CANTRAJ_BENCH cache (numpy weights, PyTorch checkpoint and
training metadata per model and seed; the timing and smoke-test runs stay out; the cache root in meta.json becomes
$CANTRAJ_BENCH). Never copies: manifests/ (published through the gated data repository), _private, predictions,
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
DRIVEDNA_DIR = os.path.dirname(GPS_DIR)              # the corpus tree that holds GPS/ (experiments/logs, hf_staging, code/)
DRIVE_DIR = os.path.dirname(DRIVEDNA_DIR)            # the authors' working root (openpilot checkout, GPS--CAN checkout)
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
           "reports/icde/tables/agent_live_check.csv", "reports/icde/tables/agent_pilot_summary.csv",
           # ceiling exploration of the agent (C-max, exploratory): report, result tables, selector validation and calibration hints
           "reports/icde/agent_ceiling.md", "reports/icde/agent_max_results.md", "reports/icde/tables/agent_max_w05.csv",
           "reports/icde/tables/agent_max_pairs_w05.csv", "reports/icde/tables/agent_max_status_w05.csv", "reports/icde/tables/agent_selector_max_val.csv",
           "reports/icde/tables/agent_max_val_hints.csv",
           # EAB supplementary material (built by scripts/build_supplementary.py; placed under supplementary/ in the release)
           "paper_icde/supplementary/supp.pdf", "paper_icde/supplementary/supp.tex", "paper_icde/supplementary/tables"]
EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", "_private", ".git"}
# machine-specific helpers and the Hugging Face staging / upload scripts are not part of the code release
EXCLUDE_FILES = {"progress.md", "artifact_readiness.md", "memwatch.sh", "stage_events_release.py", "upload_events_release.py", "patch_card_events.py",
                 "tile_pilot_maps.py", "make_pilot_report.py", "run_pilot.sh", "stop_chain2_before_full.sh",
                 "stage_public_package.py", "upload_public_package.py"}       # anonymous-package tooling: never in this (named) repository
BENCH = os.path.normpath(os.environ.get("CANTRAJ_BENCH", "/data/backup/DriveDNA-GPS_bench"))
# released with the code: the learned motion models (gru, resid, resid_nogru, gru_aug05) and the signal-set models (sig/S*),
# copied from the cache; directory names starting with these prefixes (timing and smoke-test runs) are not released
EXTRA_DIRS = {os.path.join(BENCH, "models", "learn"): "models/learn"}
EXTRA_EXCLUDE = ("_timing", "_smoke")
# after the cache root is rewritten, no absolute data or home root may remain in a released metadata file
EXTRA_GATE = ("cache root", re.compile(r"/data/backup/|/data/datasets/|/home/"))
TEXT_EXT = {".py", ".md", ".txt", ".yaml", ".yml", ".csv", ".ini", ".cfg", ".toml", ".json", ".sh"}
DATA_EXT = {".csv", ".json", ".md", ".txt"}
OPENPILOT = os.path.join(DRIVE_DIR, "openpilot")
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
    # a script that exported the openpilot checkout as the whole PYTHONPATH must still find the cantraj package in the release
    txt = txt.replace(f"export PYTHONPATH={OPENPILOT}\n", 'export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)/src${PYTHONPATH:+:$PYTHONPATH}"\n')
    txt = txt.replace("${PYTHON:-" + OPENPILOT + "/.venv/bin/python3}", "${PYTHON:-python3}")
    txt = txt.replace(f"{OPENPILOT}/.venv/bin/python3", "${PYTHON:-python3}" if ext == ".sh" else "python3")
    txt = txt.replace(f"{OPENPILOT}/.venv/bin/python", "${PYTHON:-python3}" if ext == ".sh" else "python")
    txt = txt.replace(OPENPILOT, "$OPENPILOT_DIR")
    txt = txt.replace(CITY, "the Controlled-supplement city")
    if ext == ".sh":
        txt = txt.replace("/data/backup/DriveDNA-GPS_bench", "${CANTRAJ_BENCH:-/data/backup/DriveDNA-GPS_bench}")
        txt = txt.replace(f"{DRIVEDNA_DIR}/experiments/logs", "${CANTRAJ_LOGS:-logs}")
        txt = txt.replace(f"cd {GPS_DIR}", 'cd "$(dirname "$0")/.."')
        txt = txt.replace(f"--out {DRIVE_DIR}/GPS--CAN", "--out ${RELEASE_DIR:-../GPS--CAN}")
    if rel == "scripts/m0_vehicle_params.py":
        txt = txt.replace('BENCH = "/data/backup/DriveDNA-GPS_bench"', 'BENCH = os.environ.get("CANTRAJ_BENCH", "/data/backup/DriveDNA-GPS_bench")')
        txt = txt.replace(f'DRIVES = "{DRIVEDNA_DIR}/hf_staging/index/drives.parquet"',
                          'DRIVES = os.path.join(os.environ.get("CANTRAJ_DRIVEDNA", "."), "index", "drives.parquet")')
    if rel == "scripts/m0_redecode_rlog.py":
        txt = txt.replace(f'sys.path.insert(0, "{DRIVEDNA_DIR}/code/preprocessing")',
                          'sys.path.insert(0, os.environ.get("DRIVEDNA_CODE", "code/preprocessing"))   # DriveDNA preprocessing helpers')
    # manifests referenced by absolute path -> the CANTRAJ_MANIFESTS directory; Natural Earth polygons -> environment variables
    txt = re.sub('"' + re.escape(GPS_DIR) + r'/manifests/([^"]+)"',
                 lambda m: f'os.path.join(os.environ.get("CANTRAJ_MANIFESTS", "manifests"), "{m.group(1)}")', txt)
    txt = re.sub('"' + re.escape(DRIVE_DIR) + r'/LKA_Detection/lkas_analysis_output/([a-z_]+)\.geojson"',
                 lambda m: f'os.environ.get("CANTRAJ_{m.group(1).upper()}_GEOJSON", "{m.group(1)}.geojson")   # Natural Earth polygons', txt)
    return txt


# files copied under another path in the release (the EAB supplementary material: PDF and its sources)
RENAME = {"paper_icde/supplementary/supp.pdf": "supplementary/supplementary.pdf", "paper_icde/supplementary/supp.tex": "supplementary/source/supp.tex",
          "paper_icde/supplementary/tables": "supplementary/source/tables"}


def dest_of(rel):
    for src, dst in RENAME.items():
        if rel == src:
            return dst
        if rel.startswith(src + "/"):
            return dst + rel[len(src):]
    return rel


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


def iter_extra():
    """(absolute source, destination path in the release) of the files copied from outside the code tree."""
    for src_root, dst_root in EXTRA_DIRS.items():
        if not os.path.isdir(src_root):
            sys.exit(f"{src_root} is missing: the trained weights are part of the release (set CANTRAJ_BENCH)")
        for root, dirs, files in os.walk(src_root):
            dirs[:] = sorted(d for d in dirs if not d.startswith(EXTRA_EXCLUDE) and d not in EXCLUDE_DIRS)
            for f in sorted(files):
                p = os.path.join(root, f)
                yield p, os.path.join(dst_root, os.path.relpath(p, src_root))


def transform_extra(txt):
    """Training metadata records the cache root; the release names the environment variable instead."""
    return txt.replace(BENCH, "$CANTRAJ_BENCH")


def read_extra(src):
    return transform_extra(open(src, encoding="utf-8", errors="replace").read())


def gate_extra(src):
    ext = os.path.splitext(src)[1].lower()
    if ext not in TEXT_EXT:
        return []
    txt = read_extra(src)
    hits = []
    for name, rx in GATE + [EXTRA_GATE]:
        if name == "coordinate column" and ext not in DATA_EXT:
            continue
        m = rx.search(txt)
        if m:
            hits.append((name, txt.count("\n", 0, m.start()) + 1, m.group(0)[:60]))
    return hits


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
    extras = list(iter_extra())
    bad = [(rel, h) for rel in files for h in gate(rel)] + [(dst, h) for src, dst in extras for h in gate_extra(src)]
    for rel, (name, line, frag) in bad:
        print(f"GATE HIT  {rel}:{line}  {name}: {frag!r}")
    print(f"identifier gate: {len(bad)} hits in {len(files) + len(extras)} files ({len(extras)} under models/)")
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
        dst = os.path.join(args.out, dest_of(rel))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.splitext(rel)[1].lower() in TEXT_EXT:
            with open(dst, "w", encoding="utf-8") as f:
                f.write(read_text(rel))
            shutil.copymode(os.path.join(GPS_DIR, rel), dst)
        else:
            shutil.copy2(os.path.join(GPS_DIR, rel), dst)
        n += 1
    for src, dst_rel in extras:
        dst = os.path.join(args.out, dst_rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.splitext(src)[1].lower() in TEXT_EXT:
            with open(dst, "w", encoding="utf-8") as f:
                f.write(read_extra(src))
        else:
            shutil.copy2(src, dst)
        n += 1
    print(f"exported {n} files -> {args.out}")


if __name__ == "__main__":
    main()
