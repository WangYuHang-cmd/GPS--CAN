"""Split constraints on the drive-level split table produced by scripts/m0_gps_quality.py."""
import os

import pandas as pd
import pytest

from cantraj import paths

P = os.path.join(paths.MANIFESTS, "split_drives.parquet")
pytestmark = pytest.mark.skipif(not os.path.exists(P), reason="split table not built yet")


@pytest.fixture(scope="module")
def q():
    return pd.read_parquet(P)


def test_main_split_is_driver_disjoint(q):
    per_driver = q[q.split_main.isin(["train", "val", "test", "few_shot_heldout"])].groupby("driver").split_main.nunique()
    assert (per_driver == 1).all()


def test_seen_route_keeps_train_and_test_in_the_same_family(q):
    s = q[q.split_seen_route.isin(["train", "test"])]
    fam = s.groupby(["driver", "route_family"]).split_seen_route.agg(set)
    assert all({"train", "test"} <= v for v in fam.values)         # every family used has both a train and a test trip


def test_cross_route_holds_out_whole_families(q):
    s = q[q.split_cross_route_within_driver.isin(["train", "test"])]
    fam = s.groupby(["driver", "route_family"]).split_cross_route_within_driver.nunique()
    assert (fam == 1).all()                                         # a family is entirely train or entirely test
    per_driver = s.groupby("driver").split_cross_route_within_driver.agg(set)
    assert all(v == {"train", "test"} for v in per_driver.values)   # every driver in this split has both parts


def test_cross_metro_excludes_tampa_and_separates_atlanta(q):
    assert (q.loc[q.in_tampa, "split_cross_metro"] == "excluded").all()
    assert (q.loc[q.in_atlanta, "split_cross_metro"] == "test").all()
    assert not (q.loc[q.split_cross_metro == "train", "in_atlanta"]).any()


def test_no_raw_identifiers_in_public_tables():
    import re
    for name in ("split_drives.csv", "vehicle_params.csv", "drive_platform.csv", "availability_wide.csv"):
        p = os.path.join(paths.MANIFESTS, name)
        if not os.path.exists(p):
            continue
        txt = open(p).read()
        hex_ids = [m for m in re.findall(r"\b[0-9a-f]{16}\b", txt) if re.search(r"[a-f]", m)]
        assert not hex_ids, (name, hex_ids[:3])
        assert not re.search(r"[0-9a-f]{8}--[0-9a-f]{10}", txt), name
        assert "/data/" not in txt and "/home/" not in txt and "_priv" not in txt, name
