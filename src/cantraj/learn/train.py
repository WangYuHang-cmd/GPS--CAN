"""Training loop for the learned motion models: chunk sampling with a horizon curriculum, train-only
standardisation and vocabularies, val chunk loss, checkpoints with a numpy export for the online provider."""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch

from .data import GRID_DT, LEARN_DIR, N_FEAT, chunk_targets, load_cache
from .models import MotionGRU, ResidualGRU, chunk_loss

UNK = 0


class ChunkSampler:
    def __init__(self, index: pd.DataFrame, cfg, rng, cache_dir=LEARN_DIR, speed_noise=0.0):
        self.index = index.reset_index(drop=True)
        self.cfg = cfg
        self.rng = rng
        self.cache_dir = cache_dir
        self.caches = {}
        self.speed_noise = float(speed_noise or 0.0)      # training augmentation (std of Gaussian speed noise, m/s)

    def cache(self, i):
        r = self.index.iloc[i]
        key = (r.driver, r.drive)
        if key not in self.caches:
            self.caches[key] = load_cache(r.driver, r.drive, self.cache_dir, static=self.static(r))
        return self.caches[key]

    def static(self, r):
        return dict(wheelbase_m=r.wheelbase_m, steer_ratio=r.steer_ratio, nominal_dt=r.nominal_dt, platform=r.platform, brand=r.brand)

    def sample(self, n, length_s, drives=None):
        """n chunks of length_s from anchor_ok fixes; returns a list of chunk dicts with 'row' attached."""
        out = []
        idx = np.arange(len(self.index)) if drives is None else np.asarray(drives)
        tries = 0
        while len(out) < n and tries < 20 * n:
            tries += 1
            i = int(self.rng.choice(idx))
            r = self.index.iloc[i]
            c = self.cache(i)
            ok = np.flatnonzero(c["anchor_ok"] & (c["t_ref"] + length_s <= c["t_ref"][-1]))
            if len(ok) == 0:
                continue
            ch = chunk_targets(c, int(self.rng.choice(ok)), length_s, self.static(r), speed_noise=self.speed_noise, rng=self.rng)
            if ch is None or ch["valid"].mean() < self.cfg["chunks"]["min_valid_frac"] or len(ch["x"]) < 5:
                continue
            ch["row"] = r
            out.append(ch)
        return out


def collate(chunks, norm, vocab, device, plat_drop=0.0, brand_drop=0.0, rng=None):
    B = len(chunks)
    T = max(len(c["feats"]) for c in chunks)
    R = max(len(c["x"]) for c in chunks)
    feats = np.zeros((B, T, N_FEAT), np.float32)
    xy = np.zeros((B, R, 2), np.float32)
    tref = np.zeros((B, R), np.float64)
    valid = np.zeros((B, R), bool)
    tc = np.full((B, R), np.nan, np.float32)
    theta0 = np.zeros(B, np.float32)
    tick0 = np.zeros(B, np.float64)
    hires = np.zeros(B, bool)
    plat = np.zeros(B, np.int64)
    brand = np.zeros(B, np.int64)
    for b, c in enumerate(chunks):
        n, m = len(c["feats"]), len(c["x"])
        feats[b, :n] = c["feats"]
        if n < T:
            feats[b, n:] = c["feats"][-1]
            feats[b, n:, 6] = 1.0        # padded ticks are "missing"
        xy[b, :m, 0], xy[b, :m, 1] = c["x"], c["y"]
        tref[b, :m] = c["t_ref"]
        if m < R:
            tref[b, m:] = c["t_ref"][-1]
        valid[b, :m] = c["valid"]
        tc[b, :m] = c["theta_course"]
        theta0[b] = c["theta0"]
        tick0[b] = c["ticks"][0] - GRID_DT          # start time of the first tick
        hires[b] = c["hires"]
        p = vocab["platform"].get(str(c["row"].platform), UNK)
        br = vocab["brand"].get(str(c["row"].brand), UNK)
        if rng is not None and plat_drop > 0 and rng.random() < plat_drop:
            p = UNK
        if rng is not None and brand_drop > 0 and rng.random() < brand_drop:
            br = UNK
        plat[b], brand[b] = p, br
    feats_std = (feats - norm["mean"]) / norm["std"]
    t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)  # noqa: E731
    return dict(feats_std=t(feats_std), feats_raw=t(feats), plat=t(plat, torch.long), brand=t(brand, torch.long), theta0=t(theta0),
                tick0=t(tick0, torch.float64).float(), t_ref=t(tref, torch.float64).float(), xy_ref=t(xy), valid=t(valid, torch.bool), theta_course=t(tc), hires=t(hires, torch.bool))


def build_vocab(index):
    plats = sorted(set(index.platform.dropna().astype(str)))
    brands = sorted(set(index.brand.dropna().astype(str)))
    return dict(platform={p: i + 1 for i, p in enumerate(plats)}, brand={b: i + 1 for i, b in enumerate(brands)})


def build_norm(sampler, n=300):
    chunks = sampler.sample(n, 120.0)
    f = np.concatenate([c["feats"] for c in chunks], axis=0)
    mean = f.mean(0)
    std = f.std(0)
    std[std < 1e-3] = 1.0
    for i in (6, 7, 11):                 # binary flags: keep as is
        mean[i], std[i] = 0.0, 1.0
    return dict(mean=mean.astype(np.float32), std=std.astype(np.float32))


def make_model(kind, cfg, vocab):
    n_plat, n_brand = len(vocab["platform"]) + 1, len(vocab["brand"]) + 1
    if kind == "gru":
        m = cfg["models"]["gru"]
        return MotionGRU(n_plat, n_brand, hidden=m["hidden"], emb_plat=m["emb_plat"], emb_brand=m["emb_brand"])
    if kind in ("resid", "resid_nogru"):
        m = cfg["models"][kind]
        return ResidualGRU(n_plat, n_brand, hidden=m["hidden"], use_gru=m["use_gru"])
    raise ValueError(kind)


def export_numpy(model, path):
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    np.savez(path, **sd)


def train(kind, seed, cfg, out_dir, index_path=None, epochs=None, device=None, log=print, quick=False, steps_per_epoch=None, augment_speed_noise=0.0):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    index = pd.read_parquet(index_path or os.path.join(LEARN_DIR, "index.parquet"))
    tr = index[index.fold == "train"]
    va = index[index.fold == "val"]
    vocab = build_vocab(tr)
    s_tr = ChunkSampler(tr, cfg, rng, speed_noise=augment_speed_noise)       # augmentation on training chunks only
    s_va = ChunkSampler(va, cfg, np.random.default_rng(12345))
    norm = build_norm(s_tr, 60 if quick else 300)
    model = make_model(kind, cfg, vocab).to(device)
    tcfg = cfg["train"]
    phys = [p for n, p in model.named_parameters() if n in ("g", "brand.weight", "plat.weight") and kind != "gru"]
    others = [p for n, p in model.named_parameters() if not (n in ("g", "brand.weight", "plat.weight") and kind != "gru")]
    groups = [dict(params=others, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])]
    if phys:
        groups.append(dict(params=phys, lr=tcfg["lr_physics"], weight_decay=1e-2))
    opt = torch.optim.AdamW(groups)
    E = epochs or tcfg["epochs"]
    steps_per_epoch = steps_per_epoch or (20 if quick else tcfg.get("steps_per_epoch", 150))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=E * steps_per_epoch, eta_min=tcfg["lr_min"])
    cur = cfg["chunks"]["curriculum"]
    lengths = cfg["chunks"]["lengths_s"]
    val_chunks = s_va.sample(200 if quick else tcfg["val_chunks"], 300.0) if len(va) else []
    best, hist = math.inf, []
    os.makedirs(out_dir, exist_ok=True)

    def _validate():
        model.eval()
        losses, ades = [], []
        with torch.no_grad():
            for s in range(0, len(val_chunks), tcfg["batch_size"]):
                b = collate(val_chunks[s:s + tcfg["batch_size"]], norm, vocab, device)
                l_, p_ = chunk_loss(model, b, **cfg["loss"])
                losses.append(float(l_)); ades.append(p_["ade"])
        return float(np.mean(losses)), float(np.mean(ades))

    if val_chunks:      # reference row: the untrained model (zero heads = nominal dead reckoning for the residual models, straight line for the GRU)
        vl0, vade0 = _validate()
        hist.append(dict(epoch=-1, train_loss=math.nan, val_loss=vl0, val_ade_300s=vade0, secs=0.0))
        log(f"ep -1 (init, untrained) val {vl0:.3f} val ADE(300 s chunks) {vade0:.1f} m")
    for ep in range(E):
        model.train()
        t0 = time.time()
        if ep < cur["epochs_60_only"]:
            probs = [1.0, 0.0, 0.0]
        elif ep < cur["epochs_up_to_120"]:
            probs = [0.5, 0.5, 0.0]
        else:
            probs = cur["final_probs"]
        tot, n = 0.0, 0
        for _ in range(steps_per_epoch):
            L = float(rng.choice(lengths, p=probs))
            chunks = s_tr.sample(tcfg["batch_size"], L)
            if not chunks:
                continue
            batch = collate(chunks, norm, vocab, device, tcfg["platform_dropout"], tcfg["brand_dropout"], rng)
            loss, parts = chunk_loss(model, batch, **cfg["loss"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            opt.step()
            sched.step()
            tot += float(loss); n += 1
        model.eval()
        vl, vade = math.nan, math.nan
        if val_chunks:
            vl, vade = _validate()
        hist.append(dict(epoch=ep, train_loss=tot / max(n, 1), val_loss=vl, val_ade_300s=vade, secs=time.time() - t0))
        log(f"ep {ep:2d} train {tot / max(n, 1):.3f} val {vl:.3f} val ADE(300 s chunks) {vade:.1f} m  {time.time() - t0:.0f}s")
        if val_chunks and vl < best:
            best = vl
            torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
            export_numpy(model, os.path.join(out_dir, "weights.npz"))
    if not val_chunks:
        torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
        export_numpy(model, os.path.join(out_dir, "weights.npz"))
    meta = dict(kind=kind, seed=seed, vocab=vocab, norm=dict(mean=norm["mean"].tolist(), std=norm["std"].tolist()), config=cfg, history=hist,
                n_train_drives=int(len(tr)), n_val_drives=int(len(va)), best_val_loss=best, device=device, augment_speed_noise=float(augment_speed_noise or 0.0))
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1, default=str)
    return meta
