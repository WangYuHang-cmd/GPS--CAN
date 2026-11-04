- 2026-11-04 15:20 export and release: GPS--CAN d778b4f (39 files: agent package, scripts, configs, tables, reports, tests; fast-forward push) plus a
  follow-up commit (gate allowances for the code sha in *.run_manifest.json, GPU name without the vendor word); `artifact_readiness.md`
  is now excluded from the named repository because it names the anonymous package. Package staged (`stage_public_package.py --steps code
  benchmark gate`: benchmark/agent/ with 2,129 log / result files, 45 MB; gate 0 hits; 31 GB tree) and the upload to
  CVPRICCVICML/GPS_Reconstruction started (`upload_public_package.py upload`, log `experiments/logs/upload_public_package_1104.log`);
  gate setting untouched (manual).
