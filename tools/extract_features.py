"""Feature-vector extraction utilities from time series + window arrays.

This module is intentionally lightweight and independent from CSV/split pipelines.
Given per-wafer (or per-sample) time-series observations and window ids,
it returns a fixed-length feature vector.
"""

from __future__ import annotations

import numpy as np


def _as_1d(x) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def _as_2d(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"observations must be 2D, got shape {arr.shape}")
    return arr


def _dynamics(v: np.ndarray, t: np.ndarray) -> np.ndarray:
    if len(v) < 2:
        return np.zeros(6, dtype=float)
    max_val = float(np.max(v))
    peak_idx = float(np.argmax(v))
    tr_idx = np.where(v >= 0.9 * max_val)[0]
    rising_time = float(tr_idx[0]) if len(tr_idx) > 0 else 0.0
    final_val = float(v[-1])
    initial_val = float(v[0])
    ts = 0.05 * abs(final_val) if abs(final_val) > 1e-6 else 0.05
    devs = np.abs(v - final_val)
    ns_idx = np.where(devs > ts)[0]
    settling_time = float(ns_idx[-1]) if len(ns_idx) > 0 else 0.0
    t_rel = t - t[0]
    region = float(np.sum(t_rel * v))
    return np.array([peak_idx, rising_time, final_val, initial_val, settling_time, region], dtype=float)


def _gradient_block(r_obs: np.ndarray, r_time: np.ndarray) -> np.ndarray:
    n_steps, n_sensors = r_obs.shape
    out = np.zeros((7, n_sensors), dtype=float)
    if n_steps < 3:
        return out
    t = r_time - r_time[0]
    if (t[-1] - t[0]) < 1e-6:
        return out

    for j in range(n_sensors):
        v = r_obs[:, j]
        try:
            out[0, j] = np.polyfit(t, v, 1)[0]
        except (np.linalg.LinAlgError, ValueError):
            out[0, j] = 0.0
        if n_steps >= 4:
            out[1, j] = np.mean(np.diff(v, n=2))
        diffs = np.diff(v)
        if len(diffs) == 0:
            continue
        out[2, j] = diffs[max(0, len(diffs) // 4)]
        out[3, j] = diffs[len(diffs) // 2]
        out[4, j] = diffs[min(len(diffs) - 1, 3 * len(diffs) // 4)]
        out[5, j] = np.max(np.abs(diffs))
        out[6, j] = np.mean(diffs > 0)
    return out


def _fft_block(r_obs: np.ndarray) -> np.ndarray:
    n_steps, n_sensors = r_obs.shape
    out = np.zeros((5, n_sensors), dtype=float)
    if n_steps < 8:
        return out

    for j in range(n_sensors):
        centered = r_obs[:, j] - np.mean(r_obs[:, j])
        fft_vals = np.abs(np.fft.rfft(centered))
        if len(fft_vals) < 2:
            continue
        mags = fft_vals[1:]
        if len(mags) == 0:
            continue
        sorted_mags = np.sort(mags)[::-1]
        out[0, j] = sorted_mags[0]
        if len(sorted_mags) > 1:
            out[1, j] = sorted_mags[1]
        if len(sorted_mags) > 2:
            out[2, j] = sorted_mags[2]
        out[3, j] = float(np.argmax(mags) + 1)
        half = max(1, len(mags) // 2)
        total_energy = np.sum(mags**2) + 1e-12
        out[4, j] = np.sum(mags[:half] ** 2) / total_energy
    return out


def _cross_sensor(r_obs: np.ndarray, n_fdc: int, n_oes: int) -> np.ndarray:
    out = np.zeros(7, dtype=float)
    if len(r_obs) < 5 or n_fdc <= 0 or n_oes <= 0:
        return out

    fdc = r_obs[:, :n_fdc]
    oes = r_obs[:, n_fdc:n_fdc + n_oes]

    try:
        best = []
        for oi in range(min(n_oes, 10)):
            o_col = oes[:, oi]
            if o_col.std() < 1e-8:
                continue
            for fi in range(min(n_fdc, 10)):
                f_col = fdc[:, fi]
                if f_col.std() < 1e-8:
                    continue
                corr = np.abs(np.corrcoef(o_col, f_col)[0, 1])
                if not np.isnan(corr):
                    best.append(corr)
        best.sort(reverse=True)
        for i, corr in enumerate(best[:5]):
            out[i] = corr
    except Exception:
        pass

    try:
        valid_oes = [oes[:, i] for i in range(n_oes) if oes[:, i].std() > 1e-8]
        if len(valid_oes) >= 2:
            corr_mat = np.corrcoef(np.array(valid_oes))
            upper = corr_mat[np.triu_indices_from(corr_mat, k=1)]
            out[5] = np.nanmean(np.abs(upper))
    except Exception:
        pass

    try:
        means = np.mean(np.abs(oes), axis=0)
        total = means.sum()
        if total > 1e-8:
            out[6] = np.sum(np.arange(n_oes) * means) / total
    except Exception:
        pass

    return out


def extract_feature_vector(
    observations,
    windows,
    time_axis,
    window_order: list[int] | None = None,
    *,
    n_fdc: int = 0,
    n_oes: int = 0,
    sensor_mean: np.ndarray | None = None,
    sensor_std: np.ndarray | None = None,
) -> np.ndarray:
    """Return a fixed-length vector from time-series and window ids.

    Args:
        observations: 2D array (n_steps, n_sensors)
        windows: 1D array (n_steps,) with window per step
        time_axis: 1D array (n_steps,)
        window_order: deterministic order of windows. If None, inferred from input.
        n_fdc/n_oes: optional split for cross-sensor features.
        sensor_mean/sensor_std: optional normalization stats.
    """
    obs = np.nan_to_num(_as_2d(observations), nan=0.0)
    win = _as_1d(windows)
    t = _as_1d(time_axis)

    if len(obs) != len(win) or len(obs) != len(t):
        raise ValueError("observations, windows, time_axis must have same length")

    if sensor_mean is not None and sensor_std is not None:
        obs = (obs - sensor_mean) / (sensor_std + 1e-8)
        obs = np.clip(obs, -5, 5)
    else:
        obs = np.clip(obs, -100, 100)

    num_sensors = obs.shape[1]
    order = sorted({int(v) for v in win.tolist()}) if window_order is None else list(window_order)

    feats: list[np.ndarray] = []

    # Global blocks (7 x num_sensors)
    feats.extend([
        np.mean(obs, axis=0),
        np.std(obs, axis=0),
        np.min(obs, axis=0),
        np.max(obs, axis=0),
        np.median(obs, axis=0),
        np.quantile(obs, 0.25, axis=0),
        np.quantile(obs, 0.75, axis=0),
    ])

    prev_mean = None
    for w in order:
        m = win == w
        if not m.any():
            feats.extend([np.zeros(num_sensors)] * 24)
            feats.append(np.zeros(7))
            if prev_mean is None:
                prev_mean = np.zeros(num_sensors)
            continue

        r_obs = obs[m]
        r_t = t[m]
        r_min = np.min(r_obs, axis=0)
        r_shifted = r_obs - r_min
        r_mean = np.mean(r_shifted, axis=0)

        feats.extend([
            r_mean,
            np.std(r_shifted, axis=0),
            np.max(r_shifted, axis=0),
            r_min,
            np.full(num_sensors, r_t[-1] - r_t[0] if len(r_t) > 1 else 0.0),
        ])

        if len(r_obs) >= 2:
            dyn = np.array([_dynamics(r_obs[:, j], r_t) for j in range(num_sensors)]).T
            feats.extend([row for row in dyn])
        else:
            feats.extend([np.zeros(num_sensors)] * 6)

        feats.append(r_mean - prev_mean if prev_mean is not None else np.zeros(num_sensors))
        prev_mean = np.mean(r_obs, axis=0)

        grad = _gradient_block(r_obs, r_t)
        feats.extend([row for row in grad])

        fft = _fft_block(r_obs)
        feats.extend([row for row in fft])

        feats.append(_cross_sensor(r_obs, n_fdc=n_fdc, n_oes=n_oes))

    # Trajectory shape (2 x num_sensors)
    means = [np.mean(obs[win == w], axis=0) for w in order if (win == w).any()]
    if len(means) >= 2:
        m_arr = np.array(means)
        feats.append(m_arr[-1] - m_arr[0])
        feats.append(np.std(m_arr, axis=0))
    else:
        feats.append(np.zeros(num_sensors))
        feats.append(np.zeros(num_sensors))

    return np.concatenate(feats)


def extract_wafer_features_v3(wafer: dict, window_order: list[int] | None = None, n_fdc: int = 0, n_oes: int = 0,
                              sensor_mean: np.ndarray | None = None, sensor_std: np.ndarray | None = None) -> np.ndarray:
    """Compatibility wrapper around extract_feature_vector for dict inputs.

    Expected wafer keys: observations, window, time.
    """
    windows = wafer.get("window")
    if windows is None:
        raise KeyError("wafer must include 'window'")
    return extract_feature_vector(
        observations=wafer["observations"],
        windows=windows,
        time_axis=wafer["time"],
        window_order=window_order,
        n_fdc=n_fdc,
        n_oes=n_oes,
        sensor_mean=sensor_mean,
        sensor_std=sensor_std,
    )
