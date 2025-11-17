# Transformer Thermal Aging — Stochastic Monte Carlo Simulator  
### Python + Shiny Dashboard for Distribution Transformer Reliability

This project simulates **distribution transformer thermal aging** under
**stochastic load behavior**, including:

- Household base load variability  
- EV charging with random arrival and charging durations  
- Thermostat-driven heating loads  
- Optional **demand-response (DR)** participation  
- Ambient winter temperature from real or synthetic files  

The goal is to estimate:

- **Hot-spot temperature distributions** (IEC/IEEE C57.91 thermal model)  
- **Daily loss-of-life (LOL) distributions**  
- **Probability of exceeding thermal thresholds (110°C, 120°C, 140°C)**  
- **Impact of DR measures** (ΔPeak-HST, ΔLOL%)  

This is Project 4 in a broader portfolio of energy-system models.

---

## 🚀 Features

**Monte Carlo Simulation**

- Per-household stochastic load profiles  
- EV arrival + charging randomness  
- Heating and winter ambient temperatures  
- Aggregation by transformer  
- Thermal model: top-oil & hot-spot (IEC/IEEE 57.91 simplified)  
- FAA and daily LOL calculation  

**Comparisons**

- Baseline = **no DR participation**  
- Scenario = DR participation slider  
- Automatic ΔKPIs for LOL% and peak temperatures  

**Interactive Dashboard** (Shiny for Python)

- Histogram of peak hotspot temperatures  
- Mean ± 2σ time-series band  
- CDF of peak temperatures  
- KPI summary (Baseline, Scenario, Δ)  

---

## 📦 Project Structure

transformer_thermal_aging/
├── app/
│ └── app_shiny.py # Shiny UI + Server
├── src/
│ ├── aging_model.py # IEC/IEEE thermal equations
│ ├── monte_carlo.py # Monte Carlo simulator
│ ├── simulate_transformers.py
│ ├── plots.py # Hist, band, CDF
│ └── init.py
├── data/
│ └── winter_weather_design.csv
├── tests/
│ └── test_thermal.py
├── requirements.txt
├── .gitignore
└── README.md
