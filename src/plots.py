# src/plots.py

import numpy as np
import matplotlib.pyplot as plt


def add_theta_thresholds(ax):
    """
    Add vertical guideline lines at typical hot-spot thresholds.
    """
    for thr in (110, 120, 140):
        ax.axvline(thr, linestyle="--", linewidth=1, alpha=0.6)


def plot_peak_hist(peak_hotspot_C: np.ndarray, ax=None):
    """
    Histogram of peak hot-spot temperature across Monte Carlo runs.
    """
    if ax is None:
        fig, ax = plt.subplots()

    peak_hotspot_C = np.asarray(peak_hotspot_C, dtype=float)
    ax.hist(peak_hotspot_C, bins=20)
    add_theta_thresholds(ax)

    ax.set_title("Peak Hot-Spot Temperature (°C)")
    ax.set_xlabel("°C")
    ax.set_ylabel("Count")

    return ax


def plot_mean_band(hs_timeseries_runs: np.ndarray, ax=None):
    """
    Plot mean ± 2σ band of hot-spot temperature over the day.

    hs_timeseries_runs: array of shape (n_runs, 24)
    """
    if ax is None:
        fig, ax = plt.subplots()

    hs_timeseries_runs = np.asarray(hs_timeseries_runs, dtype=float)
    mean = hs_timeseries_runs.mean(axis=0)
    std = hs_timeseries_runs.std(axis=0)
    hours = np.arange(mean.shape[0])

    ax.plot(hours, mean)
    ax.fill_between(hours, mean - 2 * std, mean + 2 * std, alpha=0.2)

    ax.set_title("Hot-Spot Temperature: mean ± 2σ")
    ax.set_xlabel("Hour")
    ax.set_ylabel("°C")

    return ax


def plot_peak_cdf(peak_hotspot_C: np.ndarray, ax=None):
    """
    Empirical CDF of peak hot-spot temperature.

    Shows P(Θ_H <= x) vs x, with threshold lines.
    """
    if ax is None:
        fig, ax = plt.subplots()

    peak_hotspot_C = np.asarray(peak_hotspot_C, dtype=float)
    x = np.sort(peak_hotspot_C)
    if x.size == 0:
        return ax

    y = np.arange(1, x.size + 1) / x.size

    ax.plot(x, y)
    add_theta_thresholds(ax)

    ax.set_title("Peak Hot-Spot — Empirical CDF")
    ax.set_xlabel("°C")
    ax.set_ylabel("P(Θ_H ≤ x)")

    return ax
