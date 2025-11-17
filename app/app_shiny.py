# app/app_shiny.py

from shiny import App, reactive, render, ui
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Use a non-interactive backend for matplotlib (good for Shiny server)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is on sys.path so `src` is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.monte_carlo import (
    run_monte_carlo_with_timeseries,
    summarize_results,
    run_compare_baseline_dr,
)
from src.plots import (
    plot_peak_hist,
    plot_mean_band,
    plot_peak_cdf,
)

# -----------------------
# UI
# -----------------------

app_ui = ui.page_fluid(
    ui.h2("Transformer Thermal Aging — Monte Carlo"),
    ui.layout_columns(
        ui.card(
            ui.input_slider("n_runs", "Monte Carlo runs", 20, 500, 100, step=10),
            ui.input_numeric("n_households", "Total households", 48),
            ui.input_numeric("hh_per_tx", "Households per transformer", 12),
            ui.input_numeric("tx_kva", "Transformer kVA", 100),
            ui.input_slider("dr", "DR participation", 0.0, 1.0, 0.3, step=0.05),
            ui.input_switch("compare", "Compare with (DR = 0) baseline", value=True),
            ui.input_action_button("go", "Run Simulation", class_="btn-primary"),
        ),
        ui.card(
            ui.h4("Summary KPIs"),
            ui.output_table("kpi_table"),
        ),
    ),
    ui.layout_columns(
        ui.card(
            ui.h4("Histogram: Peak Hot-Spot (°C)"),
            ui.output_plot("hist_plot"),
        ),
        ui.card(
            ui.h4("Time Series: mean ± 2σ"),
            ui.output_plot("band_plot"),
        ),
    ),
    ui.card(
        ui.h4("CDF: Peak Hot-Spot (°C)"),
        ui.output_plot("cdf_plot"),
    ),
    fillable=True,
)

# -----------------------
# Server
# -----------------------

def server(input, output, session):
    @reactive.event(input.go)
    def run_sim():
        """Run either a single scenario or baseline vs DR comparison."""
        args = dict(
            n_runs=int(input.n_runs()),
            n_households=int(input.n_households()),
            households_per_tx=int(input.hh_per_tx()),
            tx_kVA=float(input.tx_kva()),
            dr_participation=float(input.dr()),
            seed=1234,
        )
        if bool(input.compare()):
            return run_compare_baseline_dr(**args)
        else:
            return run_monte_carlo_with_timeseries(**args)

    @output
    @render.plot
    def hist_plot():
        res = run_sim()
        fig, ax = plt.subplots()

        if bool(input.compare()):
            # Baseline
            plot_peak_hist(res["baseline"]["peak_hotspot_C"], ax=ax)
            # DR overlay
            ax.hist(res["dr"]["peak_hotspot_C"], bins=20, alpha=0.5)
            ax.legend(["Baseline", "DR"])
        else:
            plot_peak_hist(res["peak_hotspot_C"], ax=ax)

        return fig

    @output
    @render.plot
    def band_plot():
        res = run_sim()
        fig, ax = plt.subplots()

        if bool(input.compare()):
            plot_mean_band(res["baseline"]["hs_timeseries"], ax=ax)
            plot_mean_band(res["dr"]["hs_timeseries"], ax=ax)
            ax.legend(["Baseline mean±2σ", "DR mean±2σ"])
        else:
            plot_mean_band(res["hs_timeseries"], ax=ax)

        return fig

    @output
    @render.plot
    def cdf_plot():
        res = run_sim()
        fig, ax = plt.subplots()

        if bool(input.compare()):
            plot_peak_cdf(res["baseline"]["peak_hotspot_C"], ax=ax)
            plot_peak_cdf(res["dr"]["peak_hotspot_C"], ax=ax)
            ax.legend(["Baseline", "DR"])
        else:
            plot_peak_cdf(res["peak_hotspot_C"], ax=ax)

        return fig

    @output
    @render.table
    def kpi_table():
        """
        Show summary KPIs:
        - In compare mode: Baseline, DR, and Δ rows for each metric.
        - Otherwise: just the scenario metrics.
        """
        res = run_sim()

        if bool(input.compare()):
            base = summarize_results({
                "peak_hotspot_C": res["baseline"]["peak_hotspot_C"],
                "daily_LOL_pct":  res["baseline"]["daily_LOL_pct"],
            })
            dr = summarize_results({
                "peak_hotspot_C": res["dr"]["peak_hotspot_C"],
                "daily_LOL_pct":  res["dr"]["daily_LOL_pct"],
            })

            rows = []
            for metric in ("peak_hotspot_C", "daily_LOL_pct"):
                # Baseline row
                rows.append({
                    "Metric": f"{metric} (Baseline)",
                    **{k: round(v, 3) for k, v in base[metric].items()},
                })
                # DR row
                rows.append({
                    "Metric": f"{metric} (DR)",
                    **{k: round(v, 3) for k, v in dr[metric].items()},
                })
                # Delta row: DR - Baseline
                rows.append({
                    "Metric": f"Δ {metric} (DR - Base)",
                    **{
                        k: round(dr[metric][k] - base[metric][k], 3)
                        for k in base[metric]
                    },
                })

            return pd.DataFrame(rows)

        else:
            summary = summarize_results({
                "peak_hotspot_C": res["peak_hotspot_C"],
                "daily_LOL_pct":  res["daily_LOL_pct"],
            })
            rows = []
            for metric, stats in summary.items():
                rows.append({
                    "Metric": metric,
                    **{k: round(v, 3) for k, v in stats.items()},
                })
            return pd.DataFrame(rows)


app = App(app_ui, server)