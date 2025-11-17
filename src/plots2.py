# src/plots.py

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_peak_hist(peak_hotspot_C: np.ndarray, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.hist(peak_hotspot_C, bins=20)
    ax.set_title("Peak Hot-Spot Temperature (°C)")
    ax.set_xlabel("°C")
    ax.set_ylabel("Count")
    return ax


def plot_mean_band(hs_timeseries_runs: np.ndarray, ax=None):
    """
    hs_timeseries_runs: (n_runs, 24)
    Plots mean ± 2σ band.
    """
    if ax is None:
        fig, ax = plt.subplots()

    mean = hs_timeseries_runs.mean(axis=0)
    std = hs_timeseries_runs.std(axis=0)
    hours = np.arange(mean.shape[0])

    ax.plot(hours, mean)
    ax.fill_between(hours, mean - 2*std, mean + 2*std, alpha=0.2)
    ax.set_title("Hot-Spot Temperature: mean ± 2σ")
    ax.set_xlabel("Hour")
    ax.set_ylabel("°C")
    return ax


def add_theta_thresholds(ax):
    for x in (110, 120, 140):
        ax.axvline(x, linestyle="--", alpha=0.6)

def plot_peak_hist(peak_hotspot_C: np.ndarray, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.hist(peak_hotspot_C, bins=20)
    add_theta_thresholds(ax)
    ax.set_title("Peak Hot-Spot Temperature (°C)")
    ax.set_xlabel("°C")
    ax.set_ylabel("Count")
    return ax

def plot_peak_cdf(peak_hotspot_C: np.ndarray, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    x = np.sort(peak_hotspot_C)
    y = np.arange(1, len(x)+1) / len(x)
    ax.plot(x, y)
    add_theta_thresholds(ax)
    ax.set_title("Peak Hot-Spot — Empirical CDF")
    ax.set_xlabel("°C")
    ax.set_ylabel("P(Θ_H ≤ x)")
    return ax
