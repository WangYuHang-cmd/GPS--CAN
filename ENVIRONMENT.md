# Environment and data layout

## Software

Python 3.12 with the packages in `requirements.txt` (`pip install -r requirements.txt`). The v1 tables were produced with
numpy 2.5, pandas 2.2, pyarrow 17, scipy 1.14, shapely 2.1, pyproj 3.6, matplotlib 3.10 and torch 2.12 (training only)
on Linux. Baseline inference needs no GPU; training a learned baseline takes 15 to 25 minutes on one consumer GPU.

```
pip install -r requirements.txt
pytest -q                     # 50 unit tests; the data-dependent ones skip when the manifests are absent
```

## Data

Everything bulky lives outside the repository and is located through environment variables (defaults in
`src/cantraj/paths.py` are the authors' machine):

| variable | content | source |
|---|---|---|
| `CANTRAJ_GPS` | GPS reference: `<driver>/<drive>.csv` (time_s, positions, accuracies) and `manifest.csv` | gated Hugging Face dataset `HenryYHW/DriveDNA-GPS` |
| `CANTRAJ_DRIVEDNA` | DriveDNA release root with `raw_signals/<driver>/<drive>.parquet` (10 Hz grid) and `index/drives.parquet` | Hugging Face dataset `HenryYHW/DriveDNA` |
| `CANTRAJ_EXTRA` | optional second root with the same layout (the Controlled supplement) | `HenryYHW/DriveDNA-Controlled` |
| `CANTRAJ_BENCH` | working cache: `native/<driver>/<drive>/*.parquet` (CAN / IMU event tables), `maps/tiles/`, `preds/`, `scores/`, `models/` | native tables from `DriveDNA-GPS/events/`; maps from Geofabrik; the rest is produced by the scripts |
| `CANTRAJ_MANIFESTS` | benchmark manifests (default `./manifests`) | `DriveDNA-GPS/benchmark/` |
| `CANTRAJ_LOGS` | log directory (default `./logs`) | |

Only `time_s` and the positions of the GPS files are read; the benchmark never reads device or route identifiers.
The GPS release carries no absolute time, which is sufficient: every protocol works in drive-relative seconds.

## Maps

Road graphs are cut from OpenStreetMap extracts (Geofabrik) into 0.25 degree tiles under `$CANTRAJ_BENCH/maps/tiles/`
(`manifests/map_tiles.csv` records the source extract and md5 of every tile). `scripts/fetch_maps_all.py` downloads
and cuts the extracts by region; `scripts/plan_episode_tiles.py` adds the tiles each episode's disc needs and writes
empty tiles for sea and road-less areas. Reproducing the map rows therefore needs about 30 GB of extracts and a few
hours of cutting; the no-map rows need none of it.

## Memory and runtime

Prediction runs are per-drive multiprocess jobs (`--workers`), write partial parts every 15 minutes (`--flush-min`) and
resume (`--resume`). On a 24-core / 31 GB machine shared with other jobs the authors used 3 to 6 workers; each map
worker needs 1 to 3 GB. Scoring (`scripts/score.py`) runs over drive shards with light episodes (about 2 GB per
worker). Full-corpus wall clock for v1: dead reckoning minutes per method, map particle filter about 2 days for all
windows and protocols, offline beam search about half of that.
