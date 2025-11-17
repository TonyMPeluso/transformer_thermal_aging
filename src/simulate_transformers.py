# src/simulate_transformers.py

import numpy as np
from pathlib import Path

from src.aging_model import (
    ThermalParams,
    load_winter_ambient,
    simulate_transformer_thermal_response,
    compute_loss_of_life_percent,
)
from src.simulate_households import (
    simulate_households_day,
)

from src.monte_carlo import run_monte_carlo, summarize_results


def demo_single_transformer():
    """
    Deterministic sanity test using a hand-crafted per-unit profile.
    """
    base_dir = Path(__file__).resolve().parents[1]
    weather_path = base_dir / "data" / "winter_weather_design.csv"
    ambient = load_winter_ambient(str(weather_path))

    load_pu = np.array([
        0.4, 0.35, 0.35, 0.35, 0.4, 0.5,
        0.6, 0.7, 0.75, 0.8, 0.85, 0.9,
        0.9, 0.95, 1.0, 1.05, 1.1, 1.2,
        1.25, 1.2, 1.0, 0.8, 0.6, 0.5
    ])

    params = ThermalParams()

    theta_TO, theta_H, FAA = simulate_transformer_thermal_response(
        load_pu=load_pu,
        ambient_C=ambient,
        params=params,
        dt_h=1.0,
    )

    lol_percent = compute_loss_of_life_percent(FAA, params, dt_h=1.0)

    print("=== Demo 1: Single Transformer Winter Day (Fixed Load) ===")
    print(f"Min ambient [°C]: {ambient.min():.1f}")
    print(f"Max ambient [°C]: {ambient.max():.1f}")
    print(f"Max load [p.u.]:  {load_pu.max():.2f}")
    print(f"Peak hot-spot [°C]: {theta_H.max():.2f}")
    print(f"Daily LOL [% of normal life]: {lol_percent:.8f}")
    print("Last 5 hours hot-spot [°C]:", np.round(theta_H[-5:], 2))
    print()


def assign_households_to_transformers(
    n_households: int,
    households_per_tx: int,
) -> np.ndarray:
    """
    Returns an array of shape (n_households,) with transformer IDs.
    Transformer j gets households in a simple block assignment.
    """
    n_tx = int(np.ceil(n_households / households_per_tx))
    tx_ids = np.repeat(np.arange(n_tx), households_per_tx)[:n_households]
    return tx_ids


def demo_transformer_fleet_with_households():
    """
    Demo 2:
    - Simulate stochastic households (with EV + DR).
    - Group them into transformers.
    - Compute per-unit load and thermal response for each transformer.
    """
    base_dir = Path(__file__).resolve().parents[1]
    weather_path = base_dir / "data" / "winter_weather_design.csv"
    ambient = load_winter_ambient(str(weather_path))

    # Config
    n_households = 48
    households_per_tx = 12
    tx_kVA = 100.0              # each transformer rated 100 kVA
    voltage = 1.0               # using kW ~ kVA approximation for p.u. here

    # 1) Simulate household loads [kW]
    loads_kw, households = simulate_households_day(
        n_households=n_households,
        dr_participation=0.3,
        seed=42,
    )

    # 2) Assign households to transformers
    tx_ids = assign_households_to_transformers(n_households, households_per_tx)
    n_tx = tx_ids.max() + 1

    params = ThermalParams()

    print("=== Demo 2: Fleet of Transformers with Stochastic Households ===")
    print(f"Total households: {n_households}")
    print(f"Transformers: {n_tx}, with ~{households_per_tx} households each")
    print(f"Rated transformer size: {tx_kVA:.1f} kVA\n")

    # 3) For each transformer: aggregate kW, convert to p.u., run thermal model
    for tx in range(n_tx):
        mask = (tx_ids == tx)
        tx_load_kw = loads_kw[mask, :].sum(axis=0)

        # per-unit load: assume kW ~ kVA at unity PF for this prototype
        load_pu = tx_load_kw / (tx_kVA * voltage)

        theta_TO, theta_H, FAA = simulate_transformer_thermal_response(
            load_pu=load_pu,
            ambient_C=ambient,
            params=params,
            dt_h=1.0,
        )

        lol_percent = compute_loss_of_life_percent(FAA, params, dt_h=1.0)

        print(f"Transformer {tx}:")
        print(f"  Households: {mask.sum()}")
        print(f"  Max load [kW]: {tx_load_kw.max():.2f}")
        print(f"  Max load [p.u.]: {load_pu.max():.2f}")
        print(f"  Peak hot-spot [°C]: {theta_H.max():.2f}")
        print(f"  Daily LOL [%]: {lol_percent:.8f}")
        print()

    print("Demo 2 complete.\n")


if __name__ == "__main__":
    print("About to run demo_single_transformer")
    demo_single_transformer()

    print("About to run demo_transformer_fleet_with_households")
    demo_transformer_fleet_with_households()

    # Import here to avoid circular imports during module import
    from src.monte_carlo import run_monte_carlo, summarize_results

    def demo_monte_carlo():
        print("=== Demo 3: Monte Carlo Reliability Analysis ===")
        results = run_monte_carlo(n_runs=100, n_households=48, households_per_tx=12)
        summary = summarize_results(results)
        for metric, stats in summary.items():
            print(f"\nMetric: {metric}")
            for k, v in stats.items():
                print(f"  {k:<6}: {v:8.3f}")
        print("\nMonte Carlo complete.\n")

    print("About to run demo_monte_carlo")
    demo_monte_carlo()
