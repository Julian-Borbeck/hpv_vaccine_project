#!/usr/bin/env python3
# ============================================================
# fig_growth_model_predicted_trajectories_only.py
#
# Purpose (main model figure):
#   Plot MODEL-PREDICTED (population-average) HPV first-dose coverage
#   trajectories over time by gavi_trajectory (NON-HIC countries only).
#
# Notes:
# - vax_fd_cov is HPV FIRST-DOSE coverage (%).
# - We exclude HICs from the regression sample.
# - 2022 dashed line is contextual (MICs approach rollout / transitions),
#   not interpreted causally.
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# -----------------------------
# Paths
# -----------------------------
INPUT_XLSX = Path(
    r"/Users/khaira_abdillah/Documents/dl_pro_country_comp/02_cleaned_data/"
    r"dataset_country_analysis_with_gavi_trajectory.xlsx"
)

OUT_DIR = Path(
    r"/Users/khaira_abdillah/Documents/dl_pro_country_comp/03_outputs/figs_gavi_background"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FIG = OUT_DIR / "fig_growth_model_predicted_trajectories_NONHIC.png"
OUT_DATA = OUT_DIR / "growth_model_predicted_trajectories_NONHIC.xlsx"

# -----------------------------
# Load + prep
# -----------------------------
df = pd.read_excel(INPUT_XLSX, engine="openpyxl")

needed = {"country_code", "year", "income_class", "vax_fd_cov", "gavi_trajectory"}
missing = needed - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["income_class"] = df["income_class"].astype("string").str.strip().str.upper()
df["vax_fd_cov"] = pd.to_numeric(df["vax_fd_cov"], errors="coerce")
df["gavi_trajectory"] = df["gavi_trajectory"].astype("string").str.strip()

df = df[df["year"].between(2015, 2024)].copy()
df = df.dropna(subset=["country_code", "year", "income_class", "vax_fd_cov", "gavi_trajectory"]).copy()

# NON-HIC only for regression
df = df[df["income_class"] != "H"].copy()

df["time"] = df["year"] - 2015
df["gavi_trajectory"] = df["gavi_trajectory"].astype(str)

print("\n=== Growth model sample (NON-HIC only) ===")
print("Rows:", df.shape[0])
print("Countries:", df["country_code"].nunique())
print(df.drop_duplicates("country_code")["gavi_trajectory"].value_counts().to_string())

# -----------------------------
# Fit model: random intercept + random slope (fallback to RI only)
# -----------------------------
ref = "Never Gavi (always)"
if ref not in df["gavi_trajectory"].unique():
    ref = df["gavi_trajectory"].value_counts().idxmax()

formula = f'vax_fd_cov ~ time * C(gavi_trajectory, Treatment(reference="{ref}"))'

res = None
try:
    print("\n--- Fitting Model B (random intercept + random slope) ---")
    mB = smf.mixedlm(
        formula=formula,
        data=df,
        groups=df["country_code"],
        re_formula="1 + time",
    )
    res = mB.fit(method="lbfgs", reml=False)
    print("Model B fitted. Converged:", getattr(res, "converged", "NA"))
except Exception as e:
    print("Model B failed; falling back to Model A (random intercept only).")
    print("Error:", repr(e))

if res is None:
    print("\n--- Fitting Model A (random intercept only) ---")
    mA = smf.mixedlm(
        formula=formula,
        data=df,
        groups=df["country_code"],
        re_formula="1",
    )
    res = mA.fit(method="lbfgs", reml=False)
    print("Model A fitted. Converged:", getattr(res, "converged", "NA"))

print("Reference trajectory:", ref)

# -----------------------------
# Predicted trajectories (population average)
# -----------------------------
trajectory_order = [
    "Never Gavi (always)",
    "Classic Gavi (always)",
    "Classic → MIC (graduated)",
    "Never → MIC (MICs entry)",
]
trajectories = [t for t in trajectory_order if t in df["gavi_trajectory"].unique()]
if not trajectories:
    trajectories = sorted(df["gavi_trajectory"].unique())

pred_rows = []
for g in trajectories:
    for t in range(0, 10):  # 2015..2024
        pred_rows.append({"gavi_trajectory": g, "time": t, "year": 2015 + t})

pred = pd.DataFrame(pred_rows)
pred["gavi_trajectory"] = pred["gavi_trajectory"].astype(str)

# population-average predictions
pred["pred_mean_cov"] = res.predict(pred)

pred.to_excel(OUT_DATA, index=False, engine="openpyxl")
print(f"\nSaved predicted trajectories table: {OUT_DATA}")

# -----------------------------
# Plot: model-predicted trajectories only
# -----------------------------
fig, ax = plt.subplots(figsize=(9, 5.2))

for g in trajectories:
    sub = pred[pred["gavi_trajectory"] == g].sort_values("year")
    ax.plot(sub["year"], sub["pred_mean_cov"], marker="o", linewidth=2.6, label=g)

# Context markers
ax.axvline(2022, linestyle="--", linewidth=1)      # MICs approach rollout / transitions
ax.axvspan(2020, 2021, alpha=0.08)                 # COVID disruption window (context)

ax.set_title("Model-predicted HPV first-dose coverage trajectories (non-HIC countries)")
ax.set_xlabel("Year")
ax.set_ylabel("Predicted HPV first-dose coverage (%)")
ax.set_xlim(2015, 2024)
ax.set_ylim(0, 100)

ax.grid(True, axis="y", alpha=0.25)
ax.legend(title="Gavi policy trajectory", bbox_to_anchor=(1.02, 1), loc="upper left")

note = "\n".join([
    "Notes: Predictions are population-average trajectories from a mixed-effects growth model.",
    "Sample excludes high-income (H) countries; HICs are used only as benchmark in descriptive figures.",
    "Dashed vertical line marks 2022 (MICs approach rollout / regime transitions).",
    "Shaded area marks 2020–2021 (COVID-19 disruption period).",
])
fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=9)

fig.tight_layout(rect=[0, 0.08, 1, 1])
fig.savefig(OUT_FIG, dpi=300)
plt.close(fig)

print(f"Saved figure: {OUT_FIG}")
print("DONE.")
