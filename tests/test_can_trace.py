"""CAN-Trace re-implementation on a small road graph with distinct segment lengths: stops and pedal releases at the
junctions become trace nodes, the line graph matches the true junction sequence (known start and area variants), the
1-D K-means threshold separates short pauses from stops, and a drive without a pedal signal is a failure."""
import numpy as np
import pandas as pd
import pytest

from cantraj.baselines.can_trace import CANTrace, NoMatch, kmeans_threshold
from cantraj.events import EventStream
from cantraj.geo import LocalFrame
from cantraj.inputs import EpisodeInit, EpisodeInputs, StaticParams
from cantraj.maps.graph import DriveGraph

LAT0, LON0 = 37.0, -122.0
#   D(-150,0) -- A(0,0) -- B(120,0) -- E(180,0)
#                 |          |
#               C(0,80)    F(120,-200) -- G(220,-200)
#                            |
#                          H(120,-290)
XY = {"A": (0, 0), "B": (120, 0), "C": (0, 80), "D": (-150, 0), "E": (180, 0), "F": (120, -200), "G": (220, -200),
      "Ex": (230, 0), "Gx": (220, -260), "H": (120, -290)}
WAYS = [("D", "A"), ("A", "B"), ("B", "E"), ("E", "Ex"), ("A", "C"), ("B", "F"), ("F", "G"), ("G", "Gx"), ("F", "H")]


def graph():
    fr = LocalFrame(LAT0, LON0)
    ids = {k: i for i, k in enumerate(XY)}
    nodes = []
    for k, (x, y) in XY.items():
        la, lo = fr.to_lla(np.array([float(x)]), np.array([float(y)]))
        nodes.append(dict(node_id=ids[k], lat=float(la[0]), lon=float(lo[0])))
    ways = pd.DataFrame([dict(way_id=w, highway="residential", oneway=0, junction="", maxspeed="", lanes="", name_len=0, n_nodes=2) for w in range(len(WAYS))])
    wn = pd.DataFrame([dict(way_id=w, seq=s, node_id=ids[n]) for w, (a, b) in enumerate(WAYS) for s, n in enumerate((a, b))])
    return DriveGraph(ways, wn, pd.DataFrame(nodes), pd.DataFrame(columns=["from_way", "via_node", "to_way", "restriction"]), fr)


def drive(pedal=True):
    """Start at A heading east at 8 m/s, stop 4 s at B, turn south to F, stop 4 s at F, continue east to G."""
    legs = [(120.0, 8.0), (200.0, 8.0), (100.0, 8.0)]
    t, v, gas = [], [], []
    tc = 0.0
    for k, (L, sp) in enumerate(legs):
        n = int(round(L / sp / 0.01))
        t += list(tc + 0.01 * np.arange(n)); v += [sp] * n; gas += [0.3] * n
        tc += n * 0.01
        if k < len(legs) - 1:
            n = 400                                              # 4 s standstill, pedal released
            t += list(tc + 0.01 * np.arange(n)); v += [0.0] * n; gas += [0.02] * n
            tc += n * 0.01
    cols = {"speed": np.array(v)}
    if pedal:
        cols["gas"] = np.array(gas)
    ev = EventStream(np.array(t), cols)
    inp = EpisodeInputs("G2", ev, StaticParams(), EpisodeInit(0.0, LAT0, LON0, 0.0, t_end=tc), input_layer="events")
    inp.map_graph = graph()
    return inp, tc


def test_kmeans_threshold_separates_pauses_from_stops():
    gaps = [0.1] * 50 + [0.2] * 10 + [30.0, 45.0]
    assert kmeans_threshold(gaps) == pytest.approx(0.2)


@pytest.mark.parametrize("start", ["known", "area"])
def test_matches_the_true_junction_sequence(start):
    inp, T = drive()
    q = np.arange(1.0, T, 1.0)
    m = CANTrace(start=start)
    xy, _, cands, _ = m.reconstruct(inp, q)
    # the vehicle is at B (120, 0) during the first stop and at F (120, -200) during the second
    t_b, t_f = 120 / 8.0 + 2.0, 120 / 8.0 + 4.0 + 200 / 8.0 + 2.0
    k_b, k_f = int(np.argmin(np.abs(q - t_b))), int(np.argmin(np.abs(q - t_f)))
    assert np.hypot(*(xy[k_b] - (120, 0))) < 15.0
    assert np.hypot(*(xy[k_f] - (120, -200))) < 15.0


def test_no_pedal_is_a_failure():
    inp, T = drive(pedal=False)
    with pytest.raises(NoMatch):
        CANTrace().reconstruct(inp, np.arange(1.0, T, 1.0))


def test_area_trace_without_an_edge_is_a_failure():
    """A drive with no stop and a constant pedal leaves fewer than two trace nodes after the short-edge merge: the area
    variant has nothing to match and reports a failure (it used to crash while placing an empty path)."""
    t = 0.01 * np.arange(3000)
    ev = EventStream(t, {"speed": np.full(len(t), 8.0), "gas": np.full(len(t), 0.3)})
    inp = EpisodeInputs("G2", ev, StaticParams(), EpisodeInit(0.0, LAT0, LON0, 0.0, t_end=30.0), input_layer="events")
    inp.map_graph = graph()
    q = np.arange(1.0, 30.0, 1.0)
    with pytest.raises(NoMatch, match="fewer than 2 trace nodes"):
        CANTrace(start="area").reconstruct(inp, q)
