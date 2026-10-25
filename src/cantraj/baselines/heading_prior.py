"""Heading handling for H- protocols without peeking at the reference.

`rotations(path_xy, K)` turns one dead-reckoned path (computed with an arbitrary start heading) into K whole-path
candidates rotated about the origin by 2*pi*k/K: the only legitimate way to enumerate unknown start headings.
"""
import numpy as np


def rotations(path_xy, K=36, theta_base=0.0):
    path_xy = np.asarray(path_xy, dtype=float)
    phis = theta_base + 2 * np.pi * np.arange(K) / K
    c, s = np.cos(phis), np.sin(phis)
    out = np.empty((K, len(path_xy), 2))
    out[:, :, 0] = c[:, None] * path_xy[None, :, 0] - s[:, None] * path_xy[None, :, 1]
    out[:, :, 1] = s[:, None] * path_xy[None, :, 0] + c[:, None] * path_xy[None, :, 1]
    return out, np.full(K, 1.0 / K)
