# 🚦 Transformer Thermal Aging — Stochastic Monte Carlo Simulator
## Reliability analytics for distribution transformers under electrification-era load uncertainty

This project simulates distribution transformer thermal aging under stochastic household load behavior, including:
- diverse baseload profiles
- thermostat-driven winter heating
- EV charging with random arrival times & session duration
- optional demand-response (DR) participation
- ambient winter temperature from real or synthetic datasets

The simulator computes transformer top-oil and hot-spot temperature using the IEC/IEEE C57.91 thermal model, then estimates:
- Daily loss-of-life (LOL) distributions
- Hot-spot temperature (HST) distributions
- Probability of exceeding thermal limits (110°C, 120°C, 140°C)
- Impact of DR strategies (ΔHST, ΔLOL%)

This is Project 4 in a broader portfolio of energy-system models focused on electrification, distribution planning, and reliability.

## 🌟 Why This Matters

Electrification (heat pumps, EVs, DERs) introduces highly variable, uncertain load shapes at the distribution level. Traditional deterministic planning tools cannot capture:
- random EV arrivals / clustering
- correlated heating loads during cold snaps
- household-by-household behavioral diversity

This simulator quantifies risk, not just averages—allowing utilities to understand:
- expected acceleration of transformer aging
- likelihood of hitting critical thermal thresholds
- value of DR programs in reducing peak thermal stress
- impacts of extreme weather or adoption scenarios

It is designed as a fast, modular tool for planners, reliability engineers, and researchers.

## 🚀 Key Features
### 1️⃣ Monte Carlo Household Load Simulation

Each household is modeled as an independent stochastic agent:
- Baseload drawn from diversified statistical profiles
- EV charging with random arrival time, start-of-charge SOC, and charging duration
- Heating based on thermostat deadbands & outdoor temperature
- Participation in optional DR events

Transformers aggregate these household loads to produce a probabilistic time-series of kVA demand.

### 2️⃣ Thermal Modeling (IEC/IEEE C57.91)

Implements the industry-standard model:
- Top-Oil Rise Over Ambient (TOA)
- Hot-Spot Rise Over Top-Oil (HST)
- Dynamic thermal time constants
- Winding temperature response
- Instantaneous and 24-hr Loss-of-Life (LOL) via Arrhenius aging acceleration factor

Outputs include:
- Max HST per simulated day
- Full temperature time-series
- Daily aging factor and percent LOL

### 3️⃣ KPI Computation Across Monte Carlo Runs

The simulator produces statistical distributions for:
- Peak HST
- Daily LOL (percent of design life lost)
- Probability of critical exceedances:
  - 110°C (normal aging)
  - 120°C (accelerated aging)
  - 140°C (emergency limit)

Comparisons are automatically generated for:
- Baseline scenario (no DR)
- DR scenario (user-selected participation rate)

ΔKPIs quantify the reliability benefit of DR actions.

### 4️⃣ Interactive Dashboard (Shiny for Python)

The dashboard (optional but included) provides intuitive exploration:
- Histogram: distribution of peak HST
- Time-series confidence band: mean ± 2σ
- CDF plot: probability of exceeding thresholds
- Summary table: Baseline vs DR vs Δ

This is ideal for:
- presenting results to utility planners
- exploring sensitivity cases
- scenario communication with stakeholders

## 📂 Project Structure
```
transformer_thermal_aging/
├── app/
│   └── app_shiny.py          # Shiny UI + server logic
├── src/
│   ├── aging_model.py        # IEC/IEEE thermal equations
│   ├── monte_carlo.py        # Monte Carlo simulation engine
│   ├── simulate_transformers.py
│   ├── plots.py              # Histograms, CDFs, time-series bands
│   └── __init__.py
├── data/
│   └── winter_weather_design.csv
├── tests/
│   └── test_thermal.py       # Unit tests for thermal model
├── requirements.txt
├── .gitignore
└── README.md
```

## 🧪 How It Works (Technical Flow)
1 - Generate household load profiles
- Baseload + heating + EV charging
- Randomized per run

2 - Aggregate to transformer load
- kVA time-series computed at 15-min / 1-min granularity

3 - Apply IEC/IEEE C57.91
- Compute TOA(t), HST(t), LOL(t)

4 - Repeat for N Monte Carlo runs
- Collect peak HST and LOL per iteration

5 - Compare baseline vs DR scenario
- DR reduces heating or shifts EV charging
- Compute ΔKPIs

6 - Visualize results
- Shiny dashboard plots scenarios side-by-side

## 📊 Example Outputs (descriptions)

(You can add actual images if desired)

• Peak Hot-Spot Temperature Distribution

Histogram showing how often the transformer reaches critical temperatures.

• Mean ± 2σ Temperature Band

Time-series enveloping uncertainty in heating + EV behavior.

• CDF of Peak Temperatures

Probability transformer exceeds 120°C under cold weather + EV load.

• KPI Summary
Scenario	LOL%	Peak HST	Exceedance Probability
Baseline	0.92%	128°C	32%
DR (30%)	0.51%	118°C	5%
Δ	−45%	−10°C	−27 pp

## ▶️ Running the Simulator
1. Install dependencies
```
pip install -r requirements.txt
```
2. Run a batch Monte Carlo simulation
```
python -m src.simulate_transformers
```
Outputs will be saved to /results or console depending on config.

3. Launch the Shiny Dashboard
```
shiny run --reload app/app_shiny.py
```
## 🔧 Configuration

The simulator accepts (via config or UI):
- number of households
- transformer kVA rating
- EV adoption rate
- heating type & thermostat settings
- ambient temperature file
- DR participation rate
- number of Monte Carlo runs

## 🧭 Portfolio Context

This project is part of a broader modeling suite:
- Feeder-level microsimulation
- Transformer EV load dashboard
- Investment optimization LP
- Financial risk dashboard (EVT)
- ZEV adoption mapping + census analysis

Together, they demonstrate expertise in:
- electrification modeling
- distribution planning
- thermal reliability analysis
- stochastic simulation
- optimization
- interactive dashboards for utility decision-making
- machine learning and forecasting
- climate risk analysis

## 📜 License

MIT License (see LICENSE.txt)
