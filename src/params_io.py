# src/params_io.py
import pandas as pd
from src.aging_model import ThermalParams
def params_for_kva(kva: float) -> ThermalParams:
    df = pd.read_csv("data/transformer_params.csv")
    row = df.iloc[(df["kVA"]-kva).abs().argsort().iloc[0]]
    return ThermalParams(
        delta_theta_TO_R=row.oil_rise_C,
        delta_theta_HR=row.hs_rise_C,
        tau_TO=row.tau_TO_h,
        n=row.n, m=row.m, L_norm_h=row.life_h
    )