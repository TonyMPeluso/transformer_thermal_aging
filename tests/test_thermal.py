import numpy as np
from src.aging_model import ThermalParams, simulate_transformer_thermal_response, compute_loss_of_life_percent

def test_top_oil_converges_to_ambient_when_unloaded():
    params = ThermalParams()
    ambient = np.full(24, -10.0)
    load_pu = np.zeros(24)
    theta_TO, theta_H, FAA = simulate_transformer_thermal_response(load_pu, ambient, params)
    # last few hours near ambient (tolerance 1 C)
    assert abs(theta_TO[-1] - ambient[-1]) < 1.0

def test_monotonic_with_load():
    params = ThermalParams()
    ambient = np.full(24, -10.0)
    load_low = np.full(24, 0.5)
    load_high = np.full(24, 1.0)
    _, theta_H_low, FAA_low = simulate_transformer_thermal_response(load_low, ambient, params)
    _, theta_H_high, FAA_high = simulate_transformer_thermal_response(load_high, ambient, params)
    assert theta_H_high.max() > theta_H_low.max()
    assert compute_loss_of_life_percent(FAA_high, params) > compute_loss_of_life_percent(FAA_low, params)
