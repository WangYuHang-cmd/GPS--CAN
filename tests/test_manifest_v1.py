"""Frozen manifest v1: anchors obey the frozen rule, splits agree with split_drives, mock platforms are off the main
board, and the recorded sha256 values match the files."""
import hashlib
import os

import numpy as np
import pandas as pd
import pytest
import yaml

from cantraj import paths

MAN = os.path.join(paths.MANIFESTS, "episode_manifest_v1.parquet")
CFG = os.path.join(paths.GPS_DIR, "configs", "benchmark_v1.yaml")
pytestmark = pytest.mark.skipif(not (os.path.exists(MAN) and os.path.exists(CFG)), reason="v1 manifest not frozen yet")


@pytest.fixture(scope="module")
def man():
    return pd.read_parquet(MAN)


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load(open(CFG))


def test_anchor_rule(man, cfg):
    e = man[man.eligible_main]
    a = cfg["anchor"]
    assert (e.anchor_speed_mps >= a["min_speed_mps"]).all()
    assert ((e.anchor_bearing_acc_deg > 0) & (e.anchor_bearing_acc_deg <= a["max_bearing_acc_deg"])).all()
    ub = e[e.gps_source.astype(str).str.contains("ublox")]
    assert (ub.anchor_h_acc_m < a["ublox_max_h_acc_m"]).all()
    assert (e.anchor_kind == "moving").all()


def test_splits_match_split_drives(man):
    sd = pd.read_parquet(os.path.join(paths.MANIFESTS, "split_drives.parquet")).set_index(["driver", "drive"])
    sub = man[[(a, b) in sd.index for a, b in zip(man.driver, man.drive)]]
    exp = [sd.loc[(a, b), "split_main"] for a, b in zip(sub.driver, sub.drive)]
    assert (np.array(exp) == sub.fold.values).all()


def test_mock_and_supplement_off_main_board(man):
    mock = man.platform.isna() | man.platform.astype(str).str.lower().eq("mock")
    assert not man.loc[mock, "eligible_main"].any()
    assert not man.loc[man.set == "controlled", "eligible_main"].any()
    assert (man.loc[man.eligible_main, "admission"] == "main").all()


def test_sha256_match(cfg):
    for name, digest in cfg["tables"].items():
        p = os.path.join(paths.MANIFESTS, name)
        assert hashlib.sha256(open(p, "rb").read()).hexdigest() == digest, name
    assert hashlib.sha256(open(MAN, "rb").read()).hexdigest() == cfg["episode_manifest"]["sha256"]


def test_calibrated_differs_from_nominal():
    vp = pd.read_csv(os.path.join(paths.MANIFESTS, "vehicle_params.csv"))
    cal = vp[vp.angle_offset_train_deg.notna() & vp.steer_ratio_eff_train.notna()]
    assert len(cal) >= 30
    assert (np.abs(cal.steer_ratio_eff_train - cal.steer_ratio) > 1e-6).any()
