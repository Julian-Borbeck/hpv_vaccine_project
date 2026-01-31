#!/usr/bin/env python3
# ============================================================
# fig_raw_vs_model_predicted_trajectories.py
#
# Purpose:
#   Show RAW mean HPV first-dose coverage vs MODEL-PREDICTED
#   trajectories by Gavi policy trajectory (non-HIC only).
#
# Key:
#   This script FITS the mixed model first so `res` is defined.
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

OUT_FIG = OUT_DIR / "fig_raw_vs_model_predicted_trajectories_NONHIC.png"
OUT_DATA_RAW = OUT_DIR / "raw_means_by_year_trajectory_NONHIC.xlsx"
OUT_DATA_PRED = OUT_DIR / "model_predicted_means_by_year_trajectory_NONHIC.xlsx"

# -----------------------------
# Load + prepare data
# -----------------------------
df = pd.read_excel(INPUT_XLSX, engine="openpyxl")

df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["income_class"] = df["income_class"].astype("string").str.strip().str.upper()
df["vax_fd_cov"] = pd.to_numeric(df["vax_fd_cov"], errors="coerce")
df["gavi_trajectory"] = df["gavi_trajectory"].astype("string").str.strip()

# Restrict years + non-missing
df = df[df["year"].between(2015, 2024)].copy()
df = df.dropna(subset=["country_code", "year", "income_class", "vax_fd_cov", "gavi_trajectory"]).copy()

# Non-HIC only
df = df[df["income_class"] != "H"].copy()

# Time variable
df["time"] = df["year"] - 2015

# Make sure gavi_trajectory is a plain string for patsy/statsmodels
df["gavi_trajectory"] = df["gavi_trajectory"].astype(str)

print("\n=== Sample for plotting + model (NON-HIC only) ===")
print("Rows:", df.shape[0])
print("Countries:", df["country_code"].nunique())
print(df.drop_duplicates("country_code")["gavi_trajectory"].value_counts().to_string())

# -----------------------------
# Fit the growth model (same as before)
# -----------------------------
ref = "Never Gavi (always)"
if ref not in df["gavi_trajectory"].unique():
    ref = df["gavi_trajectory"].value_counts().idxmax()

formula = f'vax_fd_cov ~ time * C(gavi_trajectory, Treatment(reference="{ref}"))'

print("\n--- Fitting Model B: random intercept + random slope (time) ---")
res = None
try:
    mB = smf.mixedlm(
        formula=formula,
        data=df,
        groups=df["country_code"],
        re_formula="1 + time",
    )
    res = mB.fit(method="lbfgs", reml=False)
    print("Model B fitted.")
except Exception as e:
    print("Model B failed, falling back to Model A (random intercept only).")
    print("Error:", repr(e))

if res is None:
    print("\n--- Fitting Model A: random intercept only ---")
    mA = smf.mixedlm(
        formula=formula,
        data=df,
        groups=df["country_code"],
        re_formula="1",
    )
    res = mA.fit(method="lbfgs", reml=False)
    print("Model A fitted.")

print("\nModel reference group:", ref)
print("Converged:", getattr(res, "converged", "NA"))

# -----------------------------
# RAW means by year & trajectory
# -----------------------------
raw_means = (
    df.groupby(["year", "gavi_trajectory"], as_index=False)
      .agg(raw_mean=("vax_fd_cov", "mean"))
)

raw_means.to_excel(OUT_DATA_RAW, index=False, engine="openpyxl")
print(f"Saved raw means: {OUT_DATA_RAW}")

# -----------------------------
# MODEL-PREDICTED means (population average / fixed effects)
# -----------------------------
# Use an explicit order (nice for legend & consistency)
trajectory_order = [
    "Never Gavi (always)",
    "Classic Gavi (always)",
    "Classic → MIC (graduated)",
    "Never → MIC (MICs entry)",
]
trajectories = [t for t in trajectory_order if t in df["gavi_trajectory"].unique()]
if not trajectories:
    trajectories = sorted(df["gavi_trajectory"].unique())

pred_grid = []
for g in trajectories:
    for t in range(0, 10):  # 2015–2024
        pred_grid.append({"gavi_trajectory": g, "time": t, "year": 2015 + t})

pred_df = pd.DataFrame(pred_grid)
pred_df["gavi_trajectory"] = pred_df["gavi_trajectory"].astype(str)

pred_df["pred_mean"] = res.predict(pred_df)

pred_df.to_excel(OUT_DATA_PRED, index=False, engine="openpyxl")
print(f"Saved model predictions: {OUT_DATA_PRED}")

# -----------------------------
# Plot: raw vs model
# -----------------------------
fig, ax = plt.subplots(figsize=(9.5, 5.6))

for g in trajectories:
    sub_raw = raw_means[raw_means["gavi_trajectory"] == g].sort_values("year")
    sub_pred = pred_df[pred_df["gavi_trajectory"] == g].sort_values("year")

    # Raw means (dashed)
    ax.plot(
        sub_raw["year"], sub_raw["raw_mean"],
        linestyle="--", marker="o", alpha=0.55, linewidth=1.6,
        label=f"{g} (raw)"
    )

    # Model predicted (solid)
    ax.plot(
        sub_pred["year"], sub_pred["pred_mean"],
        linestyle="-", linewidth=2.8,
        label=f"{g} (model)"
    )

ax.set_title("HPV first-dose coverage: raw means vs model-predicted trajectories (non-HIC countries)")
ax.set_xlabel("Year")
ax.set_ylabel("HPV first-dose coverage (%)")
ax.set_xlim(2015, 2024)
ax.set_ylim(0, 100)

# Mark COVID disruption window for context
ax.axvspan(2020, 2021, alpha=0.08)

ax.grid(True, axis="y", alpha=0.25)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Trajectory")

note = "\n".join([
    "Notes: vax_fd_cov is HPV FIRST-DOSE coverage.",
    "Dashed lines show raw group means; solid lines show model-predicted means (population average).",
    "Model is linear; edge-year predictions can slightly extrapolate beyond feasible bounds,",
    "but trends should align with observed means.",
])
fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=9)

fig.tight_layout(rect=[0, 0.08, 1, 1])
fig.savefig(OUT_FIG, dpi=300)
plt.close(fig)

print(f"\nSaved figure: {OUT_FIG}")
print("DONE.")
