"""Learned motion models (torch): causal GRU increment model and physics-residual GRU, with a differentiable
integrator shared by both. Inputs are the per-tick features of learn/data.features (already standardised by the
trainer); outputs per tick: heading rate omega (rad/s, left-positive) and a speed scale; the integrator turns them into
ENU positions from (0, 0, theta0) at 0.1 s steps and samples them at reference times.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .data import GRID_DT, N_FEAT

I_V, I_MISSING, I_MOVING, I_OMEGA_NOM = 0, 6, 7, 12     # raw (unstandardised) feature indices used by the physics


def integrate(omega, ds, theta0):
    """omega (B,T) rad/s, ds (B,T) metres per tick, theta0 (B,) -> theta (B,T), xy (B,T,2) at tick ends."""
    theta = theta0[:, None] + torch.cumsum(omega * GRID_DT, dim=1)
    th_mid = theta - 0.5 * omega * GRID_DT                       # heading at the middle of the tick (second-order accurate)
    x = torch.cumsum(ds * torch.cos(th_mid), dim=1)
    y = torch.cumsum(ds * torch.sin(th_mid), dim=1)
    return theta, torch.stack([x, y], dim=-1)


def sample_at(xy, theta, tick0, t_ref):
    """Linear interpolation of tick-end positions at reference times. xy (B,T,2); tick0 (B,) time of the first tick's
    start (positions at tick k are at time tick0 + (k+1)*dt); t_ref (B,R). Returns xy_ref (B,R,2), theta_ref (B,R)."""
    B, T, _ = xy.shape
    pos = (t_ref - tick0[:, None]) / GRID_DT - 1.0               # fractional tick-end index
    pos = pos.clamp(0.0, T - 1.0)
    i0 = pos.floor().long()
    i1 = (i0 + 1).clamp(max=T - 1)
    w = (pos - i0.float()).unsqueeze(-1)
    g0 = torch.gather(xy, 1, i0.unsqueeze(-1).expand(-1, -1, 2))
    g1 = torch.gather(xy, 1, i1.unsqueeze(-1).expand(-1, -1, 2))
    th0 = torch.gather(theta, 1, i0)
    th1 = torch.gather(theta, 1, i1)
    return g0 * (1 - w) + g1 * w, th0 * (1 - w.squeeze(-1)) + th1 * w.squeeze(-1)


class MotionGRU(nn.Module):
    """(c) causal increment model: GRU over features + platform/brand embeddings -> (speed scale, heading rate)."""

    def __init__(self, n_plat, n_brand, hidden=64, emb_plat=8, emb_brand=4, n_feat=N_FEAT):
        super().__init__()
        self.plat = nn.Embedding(n_plat, emb_plat)
        self.brand = nn.Embedding(n_brand, emb_brand)
        self.gru = nn.GRU(n_feat + emb_plat + emb_brand, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def step_inputs(self, feats_std, plat_id, brand_id):
        B, T, _ = feats_std.shape
        e = torch.cat([self.plat(plat_id), self.brand(brand_id)], dim=-1).unsqueeze(1).expand(-1, T, -1)
        return torch.cat([feats_std, e], dim=-1)

    def forward(self, feats_std, feats_raw, plat_id, brand_id, h0=None):
        out, h = self.gru(self.step_inputs(feats_std, plat_id, brand_id), h0)
        c, u = self.head(out).unbind(-1)
        gate = feats_raw[..., I_MOVING] * (1.0 - feats_raw[..., I_MISSING])
        omega = 0.6 * torch.tanh(u) * gate
        scale = torch.exp(0.1 * torch.tanh(c))
        v = feats_raw[..., I_V] * 10.0
        ds = v * scale * GRID_DT
        return omega, ds, h


class ResidualGRU(nn.Module):
    """(d) physics residual: omega = (a_p * omega_nom + b_p * v + r_t) * gate, ds = c_p * v * dt * exp(0.05 tanh s_t);
    a_p, b_p, c_p hierarchical (global + brand + platform); r_t, s_t from a small GRU. With the GRU disabled and all
    hierarchy parameters zero it is exactly the nominal bicycle model."""

    def __init__(self, n_plat, n_brand, hidden=32, n_feat=N_FEAT, use_gru=True):
        super().__init__()
        self.use_gru = use_gru
        self.g = nn.Parameter(torch.zeros(3))                 # alpha_g, beta_g, gamma_g
        self.brand = nn.Embedding(n_brand, 3)
        self.plat = nn.Embedding(n_plat, 3)
        nn.init.zeros_(self.brand.weight)
        nn.init.zeros_(self.plat.weight)
        if use_gru:
            self.gru = nn.GRU(n_feat, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 2)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def params(self, plat_id, brand_id):
        p = self.g[None, :] + self.brand(brand_id) + self.plat(plat_id)
        return torch.exp(p[:, 0]), p[:, 1], torch.exp(p[:, 2])    # a_p, b_p, c_p

    def forward(self, feats_std, feats_raw, plat_id, brand_id, h0=None):
        a, b, c = self.params(plat_id, brand_id)
        omega_nom = feats_raw[..., I_OMEGA_NOM] * 0.3
        v = feats_raw[..., I_V] * 10.0
        gate = feats_raw[..., I_MOVING] * (1.0 - feats_raw[..., I_MISSING])
        h = None
        if self.use_gru:
            out, h = self.gru(feats_std, h0)
            u, s = self.head(out).unbind(-1)
            r = 0.05 * torch.tanh(u)
            sc = torch.exp(0.05 * torch.tanh(s))
        else:
            r = torch.zeros_like(omega_nom)
            sc = torch.ones_like(omega_nom)
        omega = (a[:, None] * omega_nom + b[:, None] * v + r) * gate
        ds = c[:, None] * v * GRID_DT * sc
        self._last_r = r
        return omega, ds, h


def huber(x, delta):
    ax = x.abs()
    return torch.where(ax <= delta, 0.5 * x * x, delta * (ax - 0.5 * delta))


def chunk_loss(model, batch, w_pos=1.0, w_dtheta=0.5, w_ds=0.1, w_resid=1e-3):
    """batch: feats_std (B,T,F), feats_raw (B,T,F), plat (B,), brand (B,), theta0 (B,), tick0 (B,), t_ref (B,R),
    xy_ref (B,R,2), valid (B,R), theta_course (B,R) (NaN where unknown), hires (B,)."""
    omega, ds, _ = model(batch["feats_std"], batch["feats_raw"], batch["plat"], batch["brand"])
    theta, xy = integrate(omega, ds, batch["theta0"])
    xy_hat, th_hat = sample_at(xy, theta, batch["tick0"], batch["t_ref"])
    valid = batch["valid"]
    err = torch.linalg.norm(xy_hat - batch["xy_ref"], dim=-1)
    pos = (huber(err, 10.0) * valid).sum() / valid.sum().clamp(min=1)
    # per-second heading change where the GPS course is reliable (hires drives)
    tc = batch["theta_course"]
    ok = torch.isfinite(tc[:, 1:]) & torch.isfinite(tc[:, :-1]) & valid[:, 1:] & valid[:, :-1] & batch["hires"][:, None]
    dth_ref = torch.atan2(torch.sin(tc[:, 1:] - tc[:, :-1]), torch.cos(tc[:, 1:] - tc[:, :-1]))
    dth_hat = th_hat[:, 1:] - th_hat[:, :-1]
    dtheta = (huber(torch.where(ok, dth_hat - torch.nan_to_num(dth_ref), torch.zeros_like(dth_hat)), 0.05) * ok).sum() / ok.sum().clamp(min=1)
    # per-second chord length
    ok2 = valid[:, 1:] & valid[:, :-1]
    ds_ref = torch.linalg.norm(batch["xy_ref"][:, 1:] - batch["xy_ref"][:, :-1], dim=-1)
    ds_hat = torch.linalg.norm(xy_hat[:, 1:] - xy_hat[:, :-1], dim=-1)
    dsl = (huber(ds_hat - ds_ref, 1.0) * ok2).sum() / ok2.sum().clamp(min=1)
    loss = w_pos * pos + w_dtheta * dtheta + w_ds * dsl
    if hasattr(model, "_last_r") and getattr(model, "use_gru", False):
        loss = loss + w_resid * (model._last_r ** 2).mean()
    return loss, dict(pos=float(pos), dtheta=float(dtheta), ds=float(dsl), ade=float((err * valid).sum() / valid.sum().clamp(min=1)))
