"""Filesystem layout. Every bulk location can be overridden with an environment variable (see ENVIRONMENT.md):

  CANTRAJ_BENCH        working cache: native event tables, map tiles, predictions, scores, models
  CANTRAJ_GPS          GPS reference (the gated DriveDNA-GPS release or its private source): <driver>/<drive>.csv + manifest.csv
  CANTRAJ_DRIVEDNA     DriveDNA release root holding raw_signals/<driver>/<drive>.parquet and index/drives.parquet
  CANTRAJ_EXTRA        optional second DriveDNA root (the Controlled supplement) with the same layout
  CANTRAJ_MANIFESTS    benchmark manifests (default: <repo>/manifests)
  CANTRAJ_LOGS         log directory (default: <repo>/logs)

Code and public manifests live in the repository; defaults match the authors' machine so that nothing changes there."""
import os

GPS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DRIVEDNA = os.environ.get("CANTRAJ_DRIVEDNA", os.path.dirname(GPS_DIR))
BENCH = os.environ.get("CANTRAJ_BENCH", "/data/backup/DriveDNA-GPS_bench")
NATIVE = f"{BENCH}/native"
PRIVATE = f"{BENCH}/private"
PREDS = f"{BENCH}/preds"
SCORES = f"{BENCH}/scores"
GPS_PRIVATE = os.environ.get("CANTRAJ_GPS", "/data/backup/DriveDNA-GPS")   # trimmed GPS CSVs (only time_s and positions are used)
GPS_MANIFEST = f"{GPS_PRIVATE}/manifest.csv"
_EXTRA = os.environ.get("CANTRAJ_EXTRA", "/data/backup/DriveDNA_extra/DriveDNA")
_STAGING = os.environ.get("CANTRAJ_DRIVEDNA", f"{DRIVEDNA}/hf_staging")
RAW_SIGNALS = f"{_STAGING}/raw_signals"
RAW_SIGNALS_EXTRA = f"{_EXTRA}/raw_signals"
DRIVES_INDEX = f"{_STAGING}/index/drives.parquet"
DRIVES_INDEX_EXTRA = f"{_EXTRA}/index/drives.parquet"
MANIFESTS = os.environ.get("CANTRAJ_MANIFESTS", f"{GPS_DIR}/manifests")
REPORTS = f"{GPS_DIR}/reports"
LOGS = os.environ.get("CANTRAJ_LOGS", f"{DRIVEDNA}/experiments/logs" if "CANTRAJ_DRIVEDNA" not in os.environ else f"{GPS_DIR}/logs")


def grid_path(driver, drive):
    p = f"{RAW_SIGNALS}/{driver}/{drive}.parquet"
    return p if os.path.exists(p) else f"{RAW_SIGNALS_EXTRA}/{driver}/{drive}.parquet"


def native_dir(driver, drive):
    return f"{NATIVE}/{driver}/{drive}"


def gps_path(driver, drive):
    return f"{GPS_PRIVATE}/{driver}/{drive}.csv"
