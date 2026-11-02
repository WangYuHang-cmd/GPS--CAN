#!/usr/bin/env python3
"""Method condition table (paper Table 'tab:methods') from one static specification, shared with the provenance note
(reports/icde/input_provenance.md). Writes paper_icde/tables/methods.tex and reports/icde/tables/methods.csv.
Cells never use dashes (the paper's rule): an empty condition is written "none". The map column names the map extent a
method can reach: "OSM disc" is the episode's visible disc; OPD-II's matcher is footnoted because it is not cut to it."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from cantraj import paths  # noqa: E402

# method, source, signals, priors, map, mode (look-ahead), training data, output
ROWS = [
    ("Hold start", "baseline", "none", "$p_0$", "none", "online", "none", "position"),
    ("Straight line", "baseline", "speed", "$p_0, \\theta_0$", "none", "online", "none", "position"),
    ("Steering DR", "baseline", "speed, steering", "$p_0, \\theta_0$", "none", "online", "none / calib.$^a$", "position"),
    ("Start prior", "baseline", "speed", "$p_0, \\theta_0$", "none", "online", "train GPS (statistics)", "position"),
    ("Route retrieval", "baseline", "speed", "$p_0, \\theta_0$", "none", "online", "train GPS (routes)", "position"),
    ("Learned GRU", "baseline", "speed, steering", "$p_0, \\theta_0$", "none / OSM disc", "online", "train GPS", "position"),
    ("Learned physics residual", "baseline", "speed, steering", "$p_0, \\theta_0$", "none / OSM disc", "online", "train GPS", "position"),
    ("Map PF", "baseline", "speed (+steering)", "$p_0$ (+$\\theta_0$)", "OSM disc", "online", "none", "position"),
    ("Map beam", "baseline", "speed (+steering)", "$p_0$ (+$\\theta_0$)", "OSM disc", "offline", "none", "top-10 paths"),
    ("MacroTracking~\\cite{gazdag2023privacy}", "re-impl.", "speed, steering", "$p_0, \\theta_0$", "OSM disc", "online", "none / calib.$^a$", "position"),
    ("OPD-II~\\cite{bianchi2025opd}", "re-impl.", "speed, steering", "$p_0, \\theta_0$", "OSM, Meili$^e$", "delayed $\\le$3\\,s$^b$", "none / calib.$^a$", "position"),
    ("Elastic Pathing~\\cite{gao2017elastic}", "re-impl.", "speed (1\\,Hz)", "$p_0$$^c$ (+$\\theta_0$)", "OSM disc", "offline", "val (2 values)", "top-$n$ paths"),
    ("CAN-Trace~\\cite{lin2025cantrace}", "re-impl.", "speed, accel.\\ pedal", "area$^c$ / $p_0$", "OSM disc", "offline", "none", "top-10 paths"),
    ("Dewri et al.~\\cite{dewri2013inferring}", "re-impl.", "speed (1\\,Hz)", "$p_0$$^c$ (+$\\theta_0$)", "OSM disc", "offline", "none", "top-10 paths"),
    ("SPy~\\cite{pese2020spy}", "re-impl.", "steering", "area$^c$ / $p_0$ (+$\\theta_0$)", "OSM disc", "offline", "val (2 values)", "path"),
    ("Signal-set GRU (S1 to S4)", "study", "per set", "$p_0, \\theta_0$", "none / OSM disc", "online", "train GPS", "position"),
    ("Curvature DR (class F)", "reference", "speed, localizer curv.", "$p_0, \\theta_0$", "none / OSM disc", "online", "GPS-informed$^d$", "position"),
]


def main():
    cols = ["method", "source", "signals", "priors", "map", "mode", "training data", "output"]
    t = pd.DataFrame(ROWS, columns=cols)
    os.makedirs(os.path.join(paths.GPS_DIR, "reports", "icde", "tables"), exist_ok=True)
    t.to_csv(os.path.join(paths.GPS_DIR, "reports", "icde", "tables", "methods.csv"), index=False)
    lines = [r"\begin{table*}[t]\centering\footnotesize\setlength{\tabcolsep}{3pt}", r"\caption{Methods and the information they use. $p_0$: start position, "
             r"$\theta_0$: start heading. Online methods answer each query from the signals up to that time. Map: ``OSM disc'' is the "
             r"episode's visible disc (Section~\ref{sec:problem}). DR: dead reckoning, PF: particle filter, GRU: gated recurrent unit, "
             r"S1 to S4: signal sets (Section~\ref{sec:methods}).}\label{tab:methods}",
             r"\begin{tabular}{llllllll}\toprule", " & ".join(r"\textbf{" + c + "}" for c in cols) + r" \\\midrule"]
    for r in ROWS:
        lines.append(" & ".join(r) + r" \\")
    lines += [r"\bottomrule\end{tabular}", r"\\[2pt]\footnotesize $^a$ variant with training-fold steering calibration. "
              r"$^b$ each position is fixed when its 3\,s batch is map-matched; a strictly online read-out is also reported. "
              r"$^c$ native prior first, then our adaptations to the shared protocols; ``area'' is a 5\,km square centred on the known start. "
              r"$^d$ the localizer used GPS in the software versions of this data; reference only. "
              r"$^e$ Meili matches within its Valhalla region tiles, 200\,m around the dead-reckoned trace, and is not cut to the episode's disc.", r"\end{table*}"]
    os.makedirs(os.path.join(paths.GPS_DIR, "paper_icde", "tables"), exist_ok=True)
    open(os.path.join(paths.GPS_DIR, "paper_icde", "tables", "methods.tex"), "w").write("\n".join(lines) + "\n")
    print("written methods table")


if __name__ == "__main__":
    main()
