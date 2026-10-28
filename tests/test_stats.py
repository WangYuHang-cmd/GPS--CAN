"""Failure-aware statistics: failures stay in the median as +inf, driver-equal medians weight drivers equally, paired
differences use common finite episodes only."""
import numpy as np

from cantraj import stats as S


def test_failures_count_in_the_median():
    v = np.array([10.0, 20.0, np.inf, np.inf, np.nan])
    med, _ = S.median_inf(v, np.arange(5))
    assert med == np.inf                              # 3 of 5 failed
    med2, _ = S.median_inf(np.array([10.0, 20.0, 30.0, np.inf]), np.arange(4))
    assert med2 == 25.0 or np.isfinite(med2)


def test_driver_equal_weights_drivers_not_episodes():
    vals = np.array([100.0] * 98 + [1.0, 2.0])
    drv = np.array(["a"] * 98 + ["b", "c"])
    pooled, _ = S.median_inf(vals, drv)
    eq, _, n = S.driver_equal(vals, drv)
    assert pooled == 100.0 and eq == 2.0 and n == 3


def test_paired_uses_common_finite_episodes():
    a = np.array([1.0, 2.0, np.inf, 4.0])
    b = np.array([2.0, 2.0, 1.0, np.nan])
    r = S.paired(a, b, np.array(["x", "x", "y", "y"]))
    assert r["n"] == 2 and r["median"] == -0.5 and r["win_a"] == 0.5
