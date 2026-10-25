"""Offline whole-path beam search on the OSM drive graph (F-M protocols; Plan v0.5 appendix B1).

Hypothesis = (edge e, offset o, heading correction phi, score S, parent). The vehicle heading is the dead-reckoned
heading psi_DR(t) plus phi, re-anchored to the road per hypothesis. One step per 1 Hz reference time: advance by the
CAN distance, branch at nodes over the successors (turn log-probability from the heading at the crossing, U-turns
down-weighted, turn restrictions respected), add heading-consistency and length-consistency terms, merge hypotheses
on (edge, 10 m offset bin) keeping the best score (Viterbi), prune to the beam width with a deterministic order.
Output: Top-1 = the single highest-scoring final hypothesis back-traced (the whole-path MAP), candidates 2..K the
next-best distinct edge sequences, probabilities = softmax of scores. No random numbers anywhere.
"""
from __future__ import annotations

import math

import numpy as np

from .base import OfflineBaseline
from .dead_reckoning import SOURCE_NEEDS, UNIT

ROAD_VMAX = {"motorway": 40, "motorway_link": 30, "trunk": 35, "trunk_link": 25, "primary": 30, "primary_link": 22, "secondary": 27, "secondary_link": 20,
             "tertiary": 24, "tertiary_link": 18, "unclassified": 20, "residential": 18, "living_street": 10, "road": 20}


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def integrate_can(events, static, source, t0, t_query, offset_mode="none", use_eff_ratio=False):
    """Cumulative CAN distance s(t) and relative dead-reckoned heading psi(t) (left-positive, psi(t0) = 0) at the
    query times, by exact arc steps over the event stream (gap rule: > 0.3 s or 3 nominal periods -> hold speed, omega 0)."""
    t = events.t_s
    v = events.cols["speed"]
    thr = max(3.0 * (events.nominal_dt or 0.1), 0.3)
    om_src = None
    if source == "steer":
        off = float(static.angle_offset_train_deg or 0.0) if offset_mode == "fixed" else 0.0
        sr = static.steer_ratio_eff_train if (use_eff_ratio and static.steer_ratio_eff_train) else static.steer_ratio
        om_src = lambda vv, i: vv * math.tan(math.radians(events.cols["steer_angle"][i] - off) / sr) / static.wheelbase_m  # noqa: E731
    elif source == "can_yaw":
        u = UNIT.get(static.yawrate_unit or "rad_s", 1.0)
        om_src = lambda vv, i: (static.yawrate_sign or 1) * u * events.cols["yaw_rate_can"][i]  # noqa: E731
    elif source == "curvature":
        om_src = lambda vv, i: -vv * events.cols["curvature_loc"][i]  # noqa: E731
    elif source == "loc_yaw":
        om_src = lambda vv, i: -events.cols["yaw_rate_loc"][i]  # noqa: E731
    s_ev = np.zeros(len(t))
    psi_ev = np.zeros(len(t))
    s, psi, v_h, om_h, t_last = 0.0, 0.0, 0.0, 0.0, float(t0)
    for i in range(len(t)):
        dt = float(t[i] - t_last)
        if dt > 0:
            om = 0.0 if dt > thr else om_h
            s += v_h * dt
            psi += om * dt
        vi = float(v[i])
        if math.isfinite(vi):
            v_h = vi
        if om_src is not None:
            o = om_src(v_h, i)
            if math.isfinite(o):
                om_h = o
        t_last = float(t[i])
        s_ev[i], psi_ev[i] = s, psi
    s_q = np.interp(t_query, t, s_ev, left=0.0)
    psi_q = np.interp(t_query, t, psi_ev, left=0.0)
    # extrapolate past the last event with the held speed
    late = t_query > t[-1] if len(t) else np.zeros(len(t_query), bool)
    s_q[late] = s_ev[-1] + v_h * (t_query[late] - t[-1])
    return s_q, psi_q


class MapBeamSearch(OfflineBaseline):
    def __init__(self, source="steer", beam_width=200, K=10, sigma_heading_deg=15.0, sigma_turn_deg=20.0, uturn_weight=0.05, anchor_gain=0.3,
                 sigma_len_frac=0.02, sigma_len_min_m=10.0, dead_end_weight=0.1, start_radius_m=50.0, merge_bin_m=10.0, temperature=1.0,
                 offset_mode="none", use_eff_ratio=False):
        self.source, self.B, self.K = source, beam_width, K
        self.sig_h, self.sig_t = math.radians(sigma_heading_deg), math.radians(sigma_turn_deg)
        self.log_uturn, self.anchor_gain = math.log(uturn_weight), anchor_gain
        self.sig_len_frac, self.sig_len_min, self.log_dead = sigma_len_frac, sigma_len_min_m, math.log(dead_end_weight)
        self.start_radius, self.merge_bin, self.T = start_radius_m, merge_bin_m, temperature
        self.offset_mode, self.use_eff_ratio = offset_mode, use_eff_ratio
        self.name = f"map_beam_{source}" + ("_calib" if use_eff_ratio else "")
        self.needs = SOURCE_NEEDS[source]

    def reconstruct(self, inputs, t_query):
        g = inputs.map_graph
        if g is None:
            raise ValueError("map_beam needs a map graph")
        init, static = inputs.init, inputs.static
        t_query = np.asarray(t_query, dtype=float)
        s_q, psi_q = integrate_can(inputs.events, static, self.source, init.t0, t_query, self.offset_mode, self.use_eff_ratio)
        # ---- initial hypotheses
        radius = self.start_radius
        cands = []
        while not cands and radius <= 400:
            cands = g.nearest_edges(*init.p0_xy, radius)
            if init.theta0 is not None and cands:
                cands = [e for e in cands if abs(wrap(g.bearing[e] - init.theta0)) < math.radians(45)] or cands
            radius *= 2
        if not cands:
            cands = [int(np.argmin(g.length))]
        E, O, PHI, S, PAR = [], [], [], [], []
        for e in cands:
            off, _ = g.project(e, 0.0, 0.0)
            E.append(e); O.append(off)
            if init.theta0 is not None:
                PHI.append(init.theta0); S.append(-0.5 * (wrap(g.bearing[e] - init.theta0) / self.sig_h) ** 2)
            else:
                PHI.append(float(g.bearing[e])); S.append(0.0)
            PAR.append(-1)
        hist_e, hist_o, hist_par = [np.array(E, dtype=np.int64)], [np.array(O, dtype=float)], [np.array(PAR, dtype=np.int64)]
        E, O, PHI, S = np.array(E, dtype=np.int64), np.array(O, dtype=float), np.array(PHI, dtype=float), np.array(S, dtype=float)
        s_prev, psi_prev = 0.0, 0.0
        for q in range(len(t_query)):
            ds = float(s_q[q] - s_prev)
            dpsi = float(psi_q[q] - psi_prev)
            theta_now = PHI + psi_q[q]                        # heading of each hypothesis at this query time
            turning = abs(dpsi) > math.radians(25)
            nE, nO, nPHI, nS, nPAR = [], [], [], [], []
            for h in range(len(E)):
                e, o, phi, sc = int(E[h]), float(O[h]) + max(ds, 0.0), float(PHI[h]), float(S[h])
                th = float(theta_now[h])
                # node crossings
                guard = 0
                stack = [(e, o, phi, sc)]
                out = []
                while stack:
                    e, o, phi, sc = stack.pop()
                    L = float(g.length[e])
                    if o <= L or guard > 30:
                        out.append((e, min(o, L), phi, sc))
                        continue
                    guard += 1
                    rem = o - L
                    succ = g.succ(e)
                    if not succ:
                        out.append((e, L, phi, sc + self.log_dead))
                        continue
                    frac = (L - (o - max(ds, 1e-9))) / max(ds, 1e-9) if ds > 0 else 1.0
                    th_x = th - dpsi * (1.0 - max(min(frac, 1.0), 0.0))
                    if self.source == "none":
                        opts = [x for x in succ if not x[2]] or succ
                        lp = np.full(len(opts), -math.log(len(opts)))
                    else:
                        opts = succ
                        sc_turn = np.array([-0.5 * (wrap(g.bearing[x[0]] - th_x) / self.sig_t) ** 2 + (self.log_uturn if x[2] else 0.0) for x in opts])
                        lp = sc_turn - np.logaddexp.reduce(sc_turn)
                    for (e2, _, _), l in zip(opts, lp):
                        stack.append((int(e2), rem, phi, sc + float(l)))
                for (e2, o2, phi2, sc2) in out:
                    # heading consistency and re-anchoring
                    if self.source == "none":
                        vmax = ROAD_VMAX.get(str(g.highway[e2]), 25)
                        v_now = ds                          # metres per second at 1 Hz steps
                        sc2 += -0.5 * (max(v_now - vmax, 0.0) / 5.0) ** 2
                    else:
                        d = wrap(th - float(g.bearing[e2]))
                        sc2 += -0.5 * (d / self.sig_h) ** 2
                        phi2 = phi2 - self.anchor_gain * d
                    nE.append(e2); nO.append(o2); nPHI.append(phi2); nS.append(sc2); nPAR.append(h)
                    # length consistency: when turning, also allow taking the next junction now if it is within sigma_len
                    if turning and self.source != "none":
                        L2 = float(g.length[e2])
                        slack = L2 - o2
                        sig = max(self.sig_len_min, self.sig_len_frac * s_q[q])
                        if 0 < slack <= 2 * sig:
                            succ2 = g.succ(e2)
                            if succ2:
                                sc_turn = np.array([-0.5 * (wrap(g.bearing[x[0]] - th) / self.sig_t) ** 2 + (self.log_uturn if x[2] else 0.0) for x in succ2])
                                lp = sc_turn - np.logaddexp.reduce(sc_turn)
                                j = int(np.argmax(lp))
                                e3 = int(succ2[j][0])
                                d3 = wrap(th - float(g.bearing[e3]))
                                nE.append(e3); nO.append(0.0); nPHI.append(phi2 - self.anchor_gain * d3)
                                nS.append(sc2 + float(lp[j]) - 0.5 * (slack / sig) ** 2 - 0.5 * (d3 / self.sig_h) ** 2); nPAR.append(h)
            E, O, PHI, S, PAR = np.array(nE, dtype=np.int64), np.array(nO, dtype=float), np.array(nPHI, dtype=float), np.array(nS, dtype=float), np.array(nPAR, dtype=np.int64)
            # merge on (edge, offset bin): keep the best score
            key = E * 100000 + np.floor(O / self.merge_bin).astype(np.int64)
            order = np.lexsort((PAR, O, E, -S))                 # best score first, deterministic ties
            key_sorted = key[order]
            keep = np.ones(len(order), dtype=bool)
            keep[1:] = key_sorted[1:] != key_sorted[:-1]
            seen = {}
            sel = []
            for idx in order:
                kk = int(key[idx])
                if kk in seen:
                    continue
                seen[kk] = True
                sel.append(idx)
                if len(sel) >= self.B:
                    break
            sel = np.array(sel, dtype=np.int64)
            E, O, PHI, S, PAR = E[sel], O[sel], PHI[sel], S[sel], PAR[sel]
            hist_e.append(E.copy()); hist_o.append(O.copy()); hist_par.append(PAR.copy())
            s_prev, psi_prev = float(s_q[q]), float(psi_q[q])
        # ---- backtrace the K best distinct edge sequences
        order = np.lexsort((E, -S))
        paths, probs, seqs = [], [], []
        for idx in order:
            seq_e, seq_o = [], []
            h = int(idx)
            for step in range(len(hist_e) - 1, 0, -1):
                seq_e.append(int(hist_e[step][h])); seq_o.append(float(hist_o[step][h]))
                h = int(hist_par[step][h])
            seq_e.reverse(); seq_o.reverse()
            key = tuple(seq_e)
            if key in seqs:
                continue
            seqs.append(key)
            paths.append((seq_e, seq_o, float(S[idx])))
            if len(paths) >= self.K:
                break
        scores = np.array([p[2] for p in paths])
        pr = np.exp((scores - scores.max()) / self.T)
        pr /= pr.sum()
        T = len(t_query)
        xy = np.zeros((T, 2))
        cands_xy = np.zeros((len(paths), T, 2))
        theta = np.zeros(T)
        for k, (seq_e, seq_o, _) in enumerate(paths):
            pts = g.points(np.array(seq_e, dtype=np.int64), np.array(seq_o, dtype=float))
            cands_xy[k] = pts
        xy[:] = cands_xy[0]
        theta[:] = g.bearing[np.array(paths[0][0], dtype=np.int64)]
        return xy, theta, (cands_xy if len(paths) > 1 else None), (pr if len(paths) > 1 else None)
