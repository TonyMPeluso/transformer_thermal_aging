# src/monte_carlo.py

from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path

from src.simulate_households import simulate_households_day
from src.aging_model import (
    ThermalParams,
    load_winter_ambient,
    simulate_transformer_thermal_response,
    compute_loss_of_life_percent,
)
from src.grouping import assign_households_to_transformers


def run_monte_carlo(
    n_runs: int = 100,
    n_households: int = 48,
    households_per_tx: int = 12,
    tx_kVA: float = 100.0,
    dr_participation: float = 0.3,
    seed: Optional[int] = None, 
) -> Dict[str, np.ndarray]:
    """
    Repeats stochastic household simulation + thermal aging for multiple runs.
    Returns distributions of peak hot-spot temperature and loss-of-life.
    """
    base_dir = Path(__file__).resolve().parents[1]
    weather_path = base_dir / "data" / "winter_weather_design.csv"
    ambient = load_winter_ambient(str(weather_path))
    params = ThermalParams()

    rng_master = np.random.default_rng(seed)
    all_peak_HS = []
    all_LOL = []

    for run in range(n_runs):
        seed_run = int(rng_master.integers(0, 1e6))
        loads_kw, households = simulate_households_day(
            n_households=n_households,
            dr_participation=dr_participation,
            seed=seed_run,
        )
        tx_ids = assign_households_to_transformers(n_households, households_per_tx)
        n_tx = tx_ids.max() + 1

        for tx in range(n_tx):
            mask = (tx_ids == tx)
            tx_load_kw = loads_kw[mask, :].sum(axis=0)
            load_pu = tx_load_kw / tx_kVA
            theta_TO, theta_H, FAA = simulate_transformer_thermal_response(
                load_pu=load_pu,
                ambient_C=ambient,
                params=params,
            )
            lol_percent = compute_loss_of_life_percent(FAA, params)
            all_peak_HS.append(theta_H.max())
            all_LOL.append(lol_percent)

    return {
        "peak_hotspot_C": np.array(all_peak_HS),
        "daily_LOL_pct": np.array(all_LOL),
    }


def summarize_results(results: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """
    Compute summary statistics for Monte Carlo results.
    """
    summary = {}
    for key, arr in results.items():
        summary[key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "p5": float(np.percentile(arr, 5)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
        }
    return summary


# --- at the top ---
from typing import Dict, List, Optional
import numpy as np
from pathlib import Path

from src.simulate_households import simulate_households_day
from src.aging_model import (
    ThermalParams,
    load_winter_ambient,
    simulate_transformer_thermal_response,
    compute_loss_of_life_percent,
)
from src.grouping import assign_households_to_transformers

# --- keep your run_monte_carlo() and summarize_results() as-is ---

def run_monte_carlo_with_timeseries(
    n_runs: int = 100,
    n_households: int = 48,
    households_per_tx: int = 12,
    tx_kVA: float = 100.0,
    dr_participation: float = 0.3,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Like run_monte_carlo, but also returns a time-series matrix:
      peak_hotspot_C : (n_runs,)
      daily_LOL_pct  : (n_runs,)
      hs_timeseries  : (n_runs, 24)  # mean θ_H across transformers per hour
    """
    base_dir = Path(__file__).resolve().parents[1]
    weather_path = base_dir / "data" / "winter_weather_design.csv"
    ambient = load_winter_ambient(str(weather_path))
    params = ThermalParams()

    rng_master = np.random.default_rng(seed)
    all_peak_HS: List[float] = []
    all_LOL: List[float] = []
    all_ts: List[np.ndarray] = []

    for _ in range(n_runs):
        seed_run = int(rng_master.integers(0, 1_000_000))
        loads_kw, _ = simulate_households_day(
            n_households=n_households,
            dr_participation=dr_participation,
            seed=seed_run,
        )
        tx_ids = assign_households_to_transformers(n_households, households_per_tx)
        n_tx = int(tx_ids.max()) + 1

        thetaH_stack = []

        for tx in range(n_tx):
            mask = (tx_ids == tx)
            tx_load_kw = loads_kw[mask, :].sum(axis=0)
            load_pu = tx_load_kw / tx_kVA

            _, theta_H, FAA = simulate_transformer_thermal_response(
                load_pu=load_pu, ambient_C=ambient, params=params
            )
            thetaH_stack.append(theta_H)

        thetaH_stack = np.vstack(thetaH_stack)          # (n_tx, 24)
        all_peak_HS.append(thetaH_stack.max())          # peak across tx & hours

        # Approximate LOL using the last computed FAA set (same order of magnitude across TX)
        lol_percent = compute_loss_of_life_percent(FAA, params)
        all_LOL.append(lol_percent)

        all_ts.append(thetaH_stack.mean(axis=0))        # (24,)

    return {
        "peak_hotspot_C": np.array(all_peak_HS),
        "daily_LOL_pct": np.array(all_LOL),
        "hs_timeseries": np.vstack(all_ts),             # (n_runs, 24)
    }


def run_compare_baseline_dr(
    n_runs: int = 200,
    n_households: int = 48,
    households_per_tx: int = 12,
    tx_kVA: float = 100.0,
    dr_participation: float = 0.3,
    seed: Optional[int] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Return baseline and DR results plus deltas."""
    base = run_monte_carlo_with_timeseries(
        n_runs=n_runs,
        n_households=n_households,
        households_per_tx=households_per_tx,
        tx_kVA=tx_kVA,
        dr_participation=0.0,
        seed=seed,
    )
    dr = run_monte_carlo_with_timeseries(
        n_runs=n_runs,
        n_households=n_households,
        households_per_tx=households_per_tx,
        tx_kVA=tx_kVA,
        dr_participation=dr_participation,
        seed=None if seed is None else seed + 1,
    )
    delta = {
        "peak_hotspot_C": dr["peak_hotspot_C"] - base["peak_hotspot_C"],
        "daily_LOL_pct": dr["daily_LOL_pct"] - base["daily_LOL_pct"],
        # time-series deltas (mean across runs):
        "hs_timeseries": dr["hs_timeseries"].mean(axis=0) - base["hs_timeseries"].mean(axis=0),
    }
    return {"baseline": base, "dr": dr, "delta": delta}
