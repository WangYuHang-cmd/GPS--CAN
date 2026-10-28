#!/usr/bin/env python3
"""Signal study check: does an added signal really change the model? For each trained signal-set model, evaluate the
same validation chunks (fixed seed) with the features intact and with one signal block shuffled across chunks (pedal
columns for the pedal sets, steering columns for the steering sets); report the change in loss, in the 1 s absolute
heading-change error of the turn magnitude, and the mean absolute change of the outputs. Validation drives only.
Writes reports/icde/tables/sig_signal_use.csv."""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cantraj import paths  # noqa: E402
from cantraj.learn.sig import BLOCKS, GROUPS  # noqa: E402
from cantraj.learn.sig_train import SigGRU, SigSampler, collate, sig_loss  # noqa: E402
import train_sig as T  # noqa: E402

M = os.path.join(paths.BENCH, "models", "learn", "sig")


def main():
    cfg = yaml.safe_load(open(os.path.join(paths.GPS_DIR, "configs", "baselines", "learn.yaml")))
    ix = T.study_index()
    va = ix[ix.fold == "val"]
    rows = []
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for g in GROUPS:
        for s in (0, 1, 2):
            d = os.path.join(M, g, f"seed{s}")
            if not os.path.exists(os.path.join(d, "meta.json")):
                continue
            meta = json.load(open(os.path.join(d, "meta.json")))
            scales = {k: tuple(v) for k, v in meta["scales"].items()}
            norm = {k: np.array(v, dtype=np.float32) for k, v in meta["norm"].items()}
            vocab = meta["vocab"]
            sam = SigSampler(va, g, scales, cfg, np.random.default_rng(777))
            chunks = sam.sample(600, 300.0)
            model = SigGRU(len(vocab["platform"]) + 1, len(vocab["brand"]) + 1).to(dev)
            model.load_state_dict(torch.load(os.path.join(d, "model.pt"), map_location=dev))
            model.eval()
            blocks = [b for b in GROUPS[g] if b != "speed"]

            def evaluate(chs):
                L, out_om, out_mag = [], [], []
                with torch.no_grad():
                    for i in range(0, len(chs), 128):
                        b = collate(chs[i:i + 128], norm, vocab, dev)
                        l_, p_ = sig_loss(model, b)
                        om, _, mag = model(b["feats_std"], b["feats_raw"], b["plat"], b["brand"])
                        L.append((float(l_), p_["mag"], p_["dth"]))
                        out_om.append(om.cpu().numpy().ravel())
                        out_mag.append(mag.cpu().numpy().ravel())
                a = np.array(L)
                return a.mean(0), np.concatenate(out_om), np.concatenate(out_mag)

            base, om0, mag0 = evaluate(chunks)
            for blk in blocks:
                rng = np.random.default_rng(1)
                perm = rng.permutation(len(chunks))
                shuffled = []
                for i, c in enumerate(chunks):
                    c2 = dict(c)
                    src = chunks[perm[i]]["feats"]
                    f = c["feats"].copy()
                    n = min(len(f), len(src))
                    f[:n, BLOCKS[blk]] = src[:n, BLOCKS[blk]]
                    c2["feats"] = f
                    shuffled.append(c2)
                sh, om1, mag1 = evaluate(shuffled)
                rows.append(dict(group=g, seed=s, block=blk, loss=base[0], loss_shuffled=sh[0], mag_err=base[1], mag_err_shuffled=sh[1],
                                 dth_err=base[2], dth_err_shuffled=sh[2], mean_abs_d_omega=float(np.mean(np.abs(om1 - om0))),
                                 mean_abs_d_mag=float(np.mean(np.abs(mag1 - mag0)))))
                print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    os.makedirs(os.path.join(paths.GPS_DIR, "reports", "icde", "tables"), exist_ok=True)
    out.to_csv(os.path.join(paths.GPS_DIR, "reports", "icde", "tables", "sig_signal_use.csv"), index=False)


if __name__ == "__main__":
    main()
