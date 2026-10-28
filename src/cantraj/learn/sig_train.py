"""Training for the signal-combination study (see learn/sig.py): pedal cache aligned to the main learning cache,
brand pedal scales from the train fold, the sigma-GRU, its loss and the training loop (same budget for every set)."""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .. import io, paths
from ..events import EventStream
from ..inputs import canonicalize
from .data import GRID_DT, LEARN_DIR, MAX_AGE_S, load_cache
from .models import huber, integrate, sample_at
from .sig import BINARY, F, GROUPS, mask, sig_features
from .train import UNK, build_vocab, export_numpy

SIG_DIR = os.path.join(paths.BENCH, "learn", "sig_v1")
PEDALS = ("gas", "brake", "gas_pressed", "brake_pressed")


def pedal_cache(driver, drive, fold):
    """Zero-order-hold pedal channels on the main cache's tick grid (train / val drives only)."""
    assert fold in ("train", "val"), "the learning caches never contain test drives"
    out_p = os.path.join(SIG_DIR, "pedal", f"{driver}__{drive}.npz")
    if os.path.exists(out_p):
        return out_p
    c = np.load(os.path.join(LEARN_DIR, "grid10", f"{driver}__{drive}.npz"))
    k0, n = int(c["k0"]), len(c["speed"])
    tg = GRID_DT * np.arange(k0, k0 + n)
    cs = io.load_native(driver, drive, ("carstate",)).get("carstate")
    arrs = {p: np.full(n, np.nan, np.float32) for p in PEDALS}
    if cs is not None:
        cols = canonicalize(cs, "native", "carstate")
        have = {p: cols[p] for p in PEDALS if p in cols}
        if have:
            ev = EventStream(cs["t_s"].to_numpy(dtype=float), have)
            g = ev.causal_grid(tg, max_age_s=MAX_AGE_S)
            for p in have:
                arrs[p] = g[p].astype(np.float32)
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    np.savez_compressed(out_p, **arrs)
    return out_p


def brand_scales(index):
    """{brand: (gas p1, gas p99, brake p99 or None)} from train-fold drives with a valid accelerator signal."""
    vals = {}
    for r in index[(index.fold == "train") & (index.gas_valid)].itertuples(index=False):
        z = np.load(os.path.join(SIG_DIR, "pedal", f"{r.driver}__{r.drive}.npz"))
        g = z["gas"][np.isfinite(z["gas"])]
        b = z["brake"][np.isfinite(z["brake"])] if r.brake_valid else np.array([])
        d = vals.setdefault(str(r.brand), dict(g=[], b=[]))
        if len(g):
            d["g"].append(g[:: max(1, len(g) // 20000)])
        if len(b):
            d["b"].append(b[:: max(1, len(b) // 20000)])
    out = {}
    for br, d in vals.items():
        if not d["g"]:
            continue
        g = np.concatenate(d["g"])
        b = np.concatenate(d["b"]) if d["b"] else np.array([])
        lo, hi = float(np.percentile(g, 1)), float(np.percentile(g, 99))
        if hi - lo <= 1e-6:
            continue
        out[br] = (lo, hi, float(np.percentile(b, 99)) if len(b) and np.percentile(b, 99) > 0 else None)
    return out


class SigSampler:
    def __init__(self, index, group, scales, cfg, rng):
        self.index, self.group, self.scales, self.cfg, self.rng = index.reset_index(drop=True), group, scales, cfg, rng
        self.caches = {}
        self.m = mask(group)

    def static(self, r):
        return dict(wheelbase_m=r.wheelbase_m, steer_ratio=r.steer_ratio, nominal_dt=r.nominal_dt, platform=r.platform, brand=r.brand)

    def cache(self, i):
        r = self.index.iloc[i]
        key = (r.driver, r.drive)
        if key not in self.caches:
            c = load_cache(r.driver, r.drive)
            z = np.load(os.path.join(SIG_DIR, "pedal", f"{r.driver}__{r.drive}.npz"))
            v, st = c["speed"], c["steer"]
            prev_v = np.concatenate([[v[0]], v[:-1]])
            prev_st = np.concatenate([[st[0]], st[:-1]])
            gas = z["gas"] if r.gas_valid else np.full(len(v), np.nan, np.float32)
            brk = z["brake"] if r.brake_valid else np.full(len(v), np.nan, np.float32)
            c["feats_full"] = sig_features(v, st, gas, brk, z["gas_pressed"], z["brake_pressed"], c["age"].astype(np.float32), c["missing"],
                                           prev_v, prev_st, self.static(r), self.scales, r.nominal_dt, self.group) * self.m
            self.caches[key] = c
        return self.caches[key]

    def sample(self, n, length_s):
        out, tries = [], 0
        while len(out) < n and tries < 20 * n:
            tries += 1
            i = int(self.rng.integers(len(self.index)))
            c = self.cache(i)
            ok = np.flatnonzero(c["anchor_ok"] & (c["t_ref"] + length_s <= c["t_ref"][-1]))
            if len(ok) == 0:
                continue
            ch = chunk(c, int(self.rng.choice(ok)), length_s)
            if ch is None or ch["valid"].mean() < self.cfg["chunks"]["min_valid_frac"] or len(ch["x"]) < 5:
                continue
            ch["row"] = self.index.iloc[i]
            out.append(ch)
        return out


def chunk(c, i_start, length_s):
    t0 = float(c["t_ref"][i_start])
    t_end = t0 + length_s
    k0 = int(c["k0"])
    j0 = int(math.ceil(t0 / GRID_DT)) - k0
    j1 = int(math.floor(t_end / GRID_DT)) - k0
    if j0 < 0 or j1 >= len(c["speed"]) or j1 - j0 < 10:
        return None
    feats = c["feats_full"][j0:j1 + 1].copy()
    feats[0, 1] = 0.0                                 # first-tick differences (speed, steering rate) as online
    feats[0, 8] = 0.0
    ticks = GRID_DT * np.arange(k0 + j0, k0 + j1 + 1)
    m = (c["t_ref"] > t0) & (c["t_ref"] <= t_end)
    gamma = float(np.radians(c["lon"][i_start] - c["lon0"]) * math.sin(math.radians(float(c["lat0"]))))
    cg, sg = math.cos(gamma), math.sin(gamma)
    dx, dy = c["x"][m] - c["x"][i_start], c["y"][m] - c["y"][i_start]
    xr, yr = cg * dx + sg * dy, -sg * dx + cg * dy
    return dict(feats=feats, ticks=ticks, t_ref=c["t_ref"][m], x=xr.astype(np.float32), y=yr.astype(np.float32),
                theta0=float(c["theta_course"][i_start]) - gamma, theta_course=(c["theta_course"][m] - gamma).astype(np.float32),
                valid=np.isfinite(xr))


def collate(chunks, norm, vocab, device, plat_drop=0.0, brand_drop=0.0, rng=None):
    B, T, R = len(chunks), max(len(c["feats"]) for c in chunks), max(len(c["x"]) for c in chunks)
    feats = np.zeros((B, T, F), np.float32)
    xy = np.zeros((B, R, 2), np.float32)
    tref = np.zeros((B, R), np.float64)
    valid = np.zeros((B, R), bool)
    tc = np.full((B, R), np.nan, np.float32)
    theta0, tick0 = np.zeros(B, np.float32), np.zeros(B, np.float64)
    plat, brand = np.zeros(B, np.int64), np.zeros(B, np.int64)
    for b, c in enumerate(chunks):
        n, m = len(c["feats"]), len(c["x"])
        feats[b, :n] = c["feats"]
        if n < T:
            feats[b, n:] = c["feats"][-1]
            feats[b, n:, 3] = 1.0
        xy[b, :m, 0], xy[b, :m, 1] = c["x"], c["y"]
        tref[b, :m] = c["t_ref"]
        if m < R:
            tref[b, m:] = c["t_ref"][-1]
        valid[b, :m] = c["valid"]
        tc[b, :m] = c["theta_course"]
        theta0[b], tick0[b] = c["theta0"], c["ticks"][0] - GRID_DT
        p = vocab["platform"].get(str(c["row"].platform), UNK)
        br = vocab["brand"].get(str(c["row"].brand), UNK)
        if rng is not None and plat_drop > 0 and rng.random() < plat_drop:
            p = UNK
        if rng is not None and brand_drop > 0 and rng.random() < brand_drop:
            br = UNK
        plat[b], brand[b] = p, br
    fstd = (feats - norm["mean"]) / norm["std"]
    t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)  # noqa: E731
    return dict(feats_std=t(fstd), feats_raw=t(feats), plat=t(plat, torch.long), brand=t(brand, torch.long), theta0=t(theta0),
                tick0=t(tick0, torch.float64).float(), t_ref=t(tref, torch.float64).float(), xy_ref=t(xy), valid=t(valid, torch.bool), theta_course=t(tc))


class SigGRU(nn.Module):
    def __init__(self, n_plat, n_brand, hidden=64, emb_plat=8, emb_brand=4):
        super().__init__()
        self.plat = nn.Embedding(n_plat, emb_plat)
        self.brand = nn.Embedding(n_brand, emb_brand)
        self.gru = nn.GRU(F + emb_plat + emb_brand, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 3)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, feats_std, feats_raw, plat_id, brand_id):
        B, T, _ = feats_std.shape
        e = torch.cat([self.plat(plat_id), self.brand(brand_id)], dim=-1).unsqueeze(1).expand(-1, T, -1)
        out, _ = self.gru(torch.cat([feats_std, e], dim=-1))
        c, u, a = self.head(out).unbind(-1)
        gate = feats_raw[..., 4] * (1.0 - feats_raw[..., 3])
        omega = 0.6 * torch.tanh(u) * gate
        mag = 0.6 * torch.sigmoid(a) * gate
        ds = feats_raw[..., 0] * 10.0 * torch.exp(0.1 * torch.tanh(c)) * GRID_DT
        return omega, ds, mag


def sig_loss(model, batch, w_pos=1.0, w_dth=0.5, w_mag=0.5, w_ds=0.1):
    omega, ds, mag = model(batch["feats_std"], batch["feats_raw"], batch["plat"], batch["brand"])
    theta, xy = integrate(omega, ds, batch["theta0"])
    xy_hat, th_hat = sample_at(xy, theta, batch["tick0"], batch["t_ref"])
    A = torch.cumsum(mag * GRID_DT, dim=1)
    _, A_hat = sample_at(xy, A, batch["tick0"], batch["t_ref"])
    valid = batch["valid"]
    err = torch.linalg.norm(xy_hat - batch["xy_ref"], dim=-1)
    pos = (huber(err, 10.0) * valid).sum() / valid.sum().clamp(min=1)
    tc = batch["theta_course"]
    ok = torch.isfinite(tc[:, 1:]) & torch.isfinite(tc[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    d_ref = torch.nan_to_num(torch.atan2(torch.sin(tc[:, 1:] - tc[:, :-1]), torch.cos(tc[:, 1:] - tc[:, :-1])))
    d_hat = th_hat[:, 1:] - th_hat[:, :-1]
    a_hat = A_hat[:, 1:] - A_hat[:, :-1]
    dth = (huber(torch.where(ok, d_hat - d_ref, torch.zeros_like(d_hat)), 0.05) * ok).sum() / ok.sum().clamp(min=1)
    dmag = (huber(torch.where(ok, a_hat - d_ref.abs(), torch.zeros_like(a_hat)), 0.05) * ok).sum() / ok.sum().clamp(min=1)
    ok2 = valid[:, 1:] & valid[:, :-1]
    ds_ref = torch.linalg.norm(batch["xy_ref"][:, 1:] - batch["xy_ref"][:, :-1], dim=-1)
    ds_hat = torch.linalg.norm(xy_hat[:, 1:] - xy_hat[:, :-1], dim=-1)
    dsl = (huber(ds_hat - ds_ref, 1.0) * ok2).sum() / ok2.sum().clamp(min=1)
    loss = w_pos * pos + w_dth * dth + w_mag * dmag + w_ds * dsl
    return loss, dict(pos=float(pos), dth=float(dth), mag=float(dmag), ds=float(dsl), ade=float((err * valid).sum() / valid.sum().clamp(min=1)))


def build_norm(sampler, grp_mask, n=300):
    f = np.concatenate([c["feats"] for c in sampler.sample(n, 120.0)], axis=0)
    mean, std = f.mean(0), f.std(0)
    std[std < 1e-3] = 1.0
    mean[BINARY], std[BINARY] = 0.0, 1.0
    off = grp_mask == 0
    mean[off], std[off] = 0.0, 1.0
    return dict(mean=mean.astype(np.float32), std=std.astype(np.float32))


def train_sig(group, seed, cfg, index, scales, out_dir, epochs, steps_per_epoch, device=None, log=print):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr, va = index[index.fold == "train"], index[index.fold == "val"]
    vocab = build_vocab(tr)
    s_tr = SigSampler(tr, group, scales, cfg, rng)
    s_va = SigSampler(va, group, scales, cfg, np.random.default_rng(12345))
    norm = build_norm(s_tr, mask(group))
    n_plat, n_brand = len(vocab["platform"]) + 1, len(vocab["brand"]) + 1
    model = SigGRU(n_plat, n_brand).to(device)
    tc = cfg["train"]
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch, eta_min=tc["lr_min"])
    cur, lengths = cfg["chunks"]["curriculum"], cfg["chunks"]["lengths_s"]
    val_chunks = s_va.sample(1000, 300.0)
    os.makedirs(out_dir, exist_ok=True)
    best, hist = math.inf, []

    def _validate():
        model.eval()
        ls, ad, nl = [], [], []
        with torch.no_grad():
            for s in range(0, len(val_chunks), tc["batch_size"]):
                b = collate(val_chunks[s:s + tc["batch_size"]], norm, vocab, device)
                l_, p_ = sig_loss(model, b)
                ls.append(float(l_)); ad.append(p_["ade"]); nl.append(p_["mag"])
        return float(np.mean(ls)), float(np.mean(ad)), float(np.mean(nl))

    for ep in range(epochs):
        model.train()
        t0 = time.time()
        probs = [1.0, 0.0, 0.0] if ep < cur["epochs_60_only"] else ([0.5, 0.5, 0.0] if ep < cur["epochs_up_to_120"] else cur["final_probs"])
        tot, n = 0.0, 0
        for _ in range(steps_per_epoch):
            chunks = s_tr.sample(tc["batch_size"], float(rng.choice(lengths, p=probs)))
            if not chunks:
                continue
            batch = collate(chunks, norm, vocab, device, tc["platform_dropout"], tc["brand_dropout"], rng)
            loss, _ = sig_loss(model, batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step()
            sched.step()
            tot += float(loss); n += 1
        vl, vade, vnll = _validate()
        hist.append(dict(epoch=ep, train_loss=tot / max(n, 1), val_loss=vl, val_ade_300s=vade, val_mag=vnll, secs=time.time() - t0))
        log(f"{group} s{seed} ep {ep:2d} train {tot / max(n, 1):.3f} val {vl:.3f} ADE300 {vade:.1f} MAG {vnll:.4f} {time.time() - t0:.0f}s")
        if vl < best:
            best = vl
            torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
            export_numpy(model, os.path.join(out_dir, "weights.npz"))
    meta = dict(kind="sig", group=group, seed=seed, vocab=vocab, scales={k: list(v) for k, v in scales.items()},
                norm=dict(mean=norm["mean"].tolist(), std=norm["std"].tolist()), history=hist, best_val_loss=best,
                n_train_drives=int(len(tr)), n_val_drives=int(len(va)), epochs=epochs, steps_per_epoch=steps_per_epoch, device=device)
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1, default=str)
    return meta
