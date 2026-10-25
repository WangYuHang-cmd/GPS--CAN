"""Causal replay of the raw event stream (online protocols) and whole-episode reconstruction (offline)."""
from __future__ import annotations

import numpy as np

from . import predfile
from .episodes import Episode, ref_times


def run_online(baseline, inputs, t_refs):
    """Deliver events with logged time <= t_ref before each query; return xy (T,2), theta (T,)."""
    baseline.reset(inputs.init, inputs.static, inputs.map_graph)
    warm = getattr(inputs, "warmup", None)
    if warm is not None and len(warm):
        baseline.warmup(warm)
    ev = inputs.events
    t_refs = np.asarray(t_refs, dtype=float)
    xy = np.empty((len(t_refs), 2))
    th = np.full(len(t_refs), np.nan)
    i = 0
    for q, t in enumerate(t_refs):
        j = int(np.searchsorted(ev.t_s, t, side="right"))
        if j > i:
            baseline.observe(_slice(ev, i, j))
            i = j
        p = baseline.predict(float(t))
        xy[q] = p.xy
        if p.theta is not None:
            th[q] = p.theta
    return xy, th


def _slice(ev, i, j):
    from .events import EventStream
    return EventStream(ev.t_s[i:j], {n: v[i:j] for n, v in ev.cols.items()}, ev.nominal_dt)


def run_offline(baseline, inputs, t_refs):
    return baseline.reconstruct(inputs, np.asarray(t_refs, dtype=float))


def run_episode(baseline, inputs, ep: Episode, protocol_spec):
    """Run the right interface for the protocol; return a prediction frame (Top-1 + whole-path candidates)."""
    t = ref_times(ep)
    if protocol_spec["mode"] == "online":
        xy, th = run_online(baseline, inputs, t)
        return predfile.new_frame(ep.episode_id, t, xy, th)
    xy, th, cands, probs = run_offline(baseline, inputs, t)
    frames = [predfile.new_frame(ep.episode_id, t, xy, th, candidate_id=0, prob=1.0 if cands is None else float(probs[0]))]
    if cands is not None:
        for k in range(1, len(cands)):
            frames.append(predfile.new_frame(ep.episode_id, t, cands[k], None, candidate_id=k, prob=float(probs[k])))
    import pandas as pd
    return pd.concat(frames, ignore_index=True)
