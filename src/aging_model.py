# src/aging_model.py

from dataclasses import dataclass
from typing import Sequence, Dict, Tuple
import numpy as np
import pandas as pd


@dataclass
class ThermalParams:
    # Ambient / reference
    theta_A_ref: float = 20.0              # Reference ambient [°C]

    # Nameplate / thermal design
    delta_theta_TO_R: float = 55.0         # Top-oil rise at rated load [°C]
    delta_theta_HR: float = 30.0           # Hot-spot rise over top-oil at rated load [°C]

    # Time constants
    tau_TO: float = 3.0                    # Top-oil time constant [h]
    tau_w: float = 0.083                   # Winding time constant [h] (5 min), not used in simplified model

    # Exponents (IEC/IEEE typical)
    n: float = 0.8                         # Top-oil exponent
    m: float = 1.6                         # Hot-spot exponent

    # Top-oil gain factor (simple approximation)
    K_TO: float = 55.0                     # Tuned so θ_TO ≈ θ_A + Δθ_TO_R at 1.0 p.u.

    # Aging / insulation life
    L_norm_h: float = 180_000.0            # Normal insulation life [h]

    # Hot-spot thresholds
    hs_limits: Dict[str, float] = None     # e.g. {"normal": 110, "alarm": 120, "emergency": 140}

    def __post_init__(self):
        if self.hs_limits is None:
            self.hs_limits = {
                "normal": 110.0,
                "alarm": 120.0,
                "emergency": 140.0,
            }


def load_winter_ambient(path: str) -> np.ndarray:
    """
    Load winter ambient temperatures [°C] from CSV with columns: hour, T_out_C.
    Returns a length-24 numpy array.
    """
    df = pd.read_csv(path)
    if "T_out_C" not in df.columns:
        raise ValueError("Expected column 'T_out_C' in winter weather file.")
    if df.shape[0] != 24:
        raise ValueError(f"Expected 24 rows (0-23h); got {df.shape[0]}.")
    return df["T_out_C"].to_numpy(dtype=float)


def aging_acceleration_factor(theta_H_C: float) -> float:
    """
    Arrhenius-type aging acceleration factor based on hot-spot temperature [°C].
    Reference ~110°C (383 K).
    """
    return float(np.exp(15000.0 / 383.0 - 15000.0 / (273.0 + theta_H_C)))


def simulate_transformer_thermal_response(
    load_pu: Sequence[float],
    ambient_C: Sequence[float],
    params: ThermalParams,
    dt_h: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Core thermal model for a single transformer.

    Inputs:
        load_pu   : per-unit load series (P_load / P_rated), len T
        ambient_C : ambient [°C], len T
        params    : ThermalParams
        dt_h      : time step [h]

    Returns:
        theta_TO : top-oil temp [°C], len T
        theta_H  : hot-spot temp [°C], len T
        FAA      : aging acceleration factor, len T
    """
    load_pu = np.asarray(load_pu, dtype=float)
    ambient_C = np.asarray(ambient_C, dtype=float)

    if load_pu.shape != ambient_C.shape:
        raise ValueError("load_pu and ambient_C must have the same length.")

    T = len(load_pu)
    theta_TO = np.zeros(T)
    theta_H = np.zeros(T)
    FAA = np.zeros(T)

    # Initialize: conservative starting point (rated rise at t=0)
    theta_TO[0] = ambient_C[0] + params.delta_theta_TO_R
    P0 = max(load_pu[0], 0.0)
    theta_H[0] = theta_TO[0] + params.delta_theta_HR * (P0 ** params.m)
    FAA[0] = aging_acceleration_factor(theta_H[0])

    alpha_TO = np.exp(-dt_h / params.tau_TO)

    for t in range(T - 1):
        P = max(load_pu[t], 0.0)
        theta_A = ambient_C[t]

        # Top-oil dynamics
        theta_TO[t + 1] = (
            theta_A
            + (theta_TO[t] - theta_A) * alpha_TO
            + params.K_TO * (P ** params.n)
        )

        # Hot-spot
        theta_H[t + 1] = theta_TO[t + 1] + params.delta_theta_HR * (P ** params.m)

        # Aging factor
        FAA[t + 1] = aging_acceleration_factor(theta_H[t + 1])

    return theta_TO, theta_H, FAA


def compute_loss_of_life_percent(
    FAA: Sequence[float],
    params: ThermalParams,
    dt_h: float = 1.0,
) -> float:
    """
    LOL_% = (sum_t FAA(t) * dt) / L_norm * 100
    """
    FAA = np.asarray(FAA, dtype=float)
    equivalent_life_h = FAA.sum() * dt_h
    return (equivalent_life_h / params.L_norm_h) * 100.0

