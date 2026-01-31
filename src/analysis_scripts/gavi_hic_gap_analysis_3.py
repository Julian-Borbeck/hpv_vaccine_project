#!/usr/bin/env python3
# ============================================================
# growth_model_hpv_first_dose_by_gavi_trajectory.py
#
# Purpose:
#   Mixed-effects "growth model" for HPV FIRST-DOSE coverage (vax_fd_cov)
#   over time (2015–2024), comparing trend slopes by gavi_trajectory.
#
# Key choices:
#   - Restrict analysis to NON-HIC countries (income_class != "H")
#     so HICs are not both benchmark and comparison group.
#   - Year is centered at 2015 (time = year - 2015) for interpretability.
#   - Reference group is "Never Gavi (always)" among non-HICs (if present).
#
# Outputs:
#   - Console summaries
#   - Excel table of fixed effects and implied slopes (optional)
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import math
from scipy.stats import norm

import statsmodels.formula.api as smf

# -----------------------------
# Paths
# -----------------------------
INPUT_XLSX = Path(
    r"/Users/khaira_abdillah/Documents/dl_pro_country_comp/02_cleaned_data/"
    r"dataset_country_analysis_with_gavi_trajectory.xlsx"
)

OUT_DIR = Path(
    r"/Users/khaira_abdillah/Documents/dl_pro_country_comp/03_outputs/model_outputs"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FE = OUT_DIR / "growth_model_fixed_effects.xlsx"
OUT_SLOPES = OUT_DIR / "growth_model_group_slopes.xlsx"

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

# Keep analysis years + non-missing coverage
df = df[df["year"].between(2015, 2024)].copy()
df = df.dropna(subset=["country_code", "year", "income_class", "vax_fd_cov", "gavi_trajectory"]).copy()

# Exclude HIC from modeling sample
df = df[df["income_class"] != "H"].copy()

# Time variable: years since 2015
df["time"] = df["year"] - 2015

# Set trajectory reference category (important!)
ref = "Never Gavi (always)"
if ref not in df["gavi_trajectory"].unique():
    # If for some reason it's absent in non-HIC sample, use the most common category
    ref = df["gavi_trajectory"].value_counts().idxmax()

df["gavi_trajectory"] = pd.Categorical(df["gavi_trajectory"])
df["gavi_trajectory"] = df["gavi_trajectory"].cat.reorder_categories(
    [ref] + [c for c in df["gavi_trajectory"].cat.categories if c != ref],
    ordered=True
)

print("\n=== Modeling sample (NON-HIC only) ===")
print("Rows (country-years):", df.shape[0])
print("Unique countries:", df["country_code"].nunique())
print("Trajectory counts (countries):")
print(df.drop_duplicates("country_code")["gavi_trajectory"].value_counts().to_string())
print("\nReference category:", ref)

# -----------------------------
# Model A: Random intercept growth model
# -----------------------------
# Fixed effects: time + trajectory + time*trajectory
# Random effects: intercept by country
df["gavi_trajectory"] = df["gavi_trajectory"].astype(str)

ref = "Never Gavi (always)"
if ref not in df["gavi_trajectory"].unique():
    ref = df["gavi_trajectory"].value_counts().idxmax()

# IMPORTANT: embed reference in the formula string
formula = f'vax_fd_cov ~ time * C(gavi_trajectory, Treatment(reference="{ref}"))'



print("\n--- Fitting Model A: random intercept ---")
mA = smf.mixedlm(
    formula=formula,
    data=df,
    groups=df["country_code"],
    re_formula="1",
)
resA = mA.fit(method="lbfgs", reml=False)

print(resA.summary())

# -----------------------------
# Model B (optional): Random intercept + random slope for time
# -----------------------------
# This can fail to converge sometimes; we try and catch errors.
print("\n--- Fitting Model B: random intercept + random slope (time) ---")
resB = None
try:
    mB = smf.mixedlm(
        formula=formula,
        data=df,
        groups=df["country_code"],
        re_formula="1 + time",
    )
    resB = mB.fit(method="lbfgs", reml=False)
    print(resB.summary())
except Exception as e:
    print("Model B failed to fit (this can happen with random slopes).")
    print("Error:", repr(e))

# Choose which result to use for downstream slope calculations
res = resB if resB is not None else resA
model_name = "Model B (RI+RS)" if resB is not None else "Model A (RI)"
print(f"\nUsing {model_name} for implied slope calculations.")

# -----------------------------
# Extract fixed effects table
# -----------------------------
fe = res.fe_params
se = res.bse_fe
fe_table = pd.DataFrame({"coef": fe, "se": se})
fe_table["z"] = fe_table["coef"] / fe_table["se"]
fe_table["p_approx"] = 2 * (1 - norm.cdf(np.abs(fe_table["z"])))
fe_table.index.name = "term"

print("\n=== Fixed effects (key terms) ===")
print(fe_table.to_string())

# Save fixed effects
fe_table.to_excel(OUT_FE, engine="openpyxl")
print(f"\nSaved fixed effects table: {OUT_FE}")

# -----------------------------
# Compute implied group-specific slopes (pp per year)
# -----------------------------
# In this model:
#   slope(reference group) = coef(time)
#   slope(other group) = coef(time) + coef(time:C(group))
#
# Build a tidy slope table for each category.
trajectory_order = [
    "Never Gavi (always)",
    "Classic Gavi (always)",
    "Classic → MIC (graduated)",
    "Never → MIC (MICs entry)",
]
cats = [c for c in trajectory_order if c in df["gavi_trajectory"].unique()]
if ref not in cats:
    cats = [ref] + [c for c in cats if c != ref]

base_slope = fe.get("time", np.nan)

rows = []
for g in cats:
    if g == ref:
        slope = base_slope
        slope_term = "time"
    else:
        # statsmodels names interaction terms like:
        # time:C(gavi_trajectory, Treatment(reference=ref))[T.<group>]
        key = f"time:C(gavi_trajectory, Treatment(reference=ref))[T.{g}]"
        slope = base_slope + fe.get(key, 0.0)
        slope_term = f"time + {key}"

    rows.append({
        "trajectory": g,
        "slope_pp_per_year": slope,
        "slope_definition": slope_term
    })

slopes = pd.DataFrame(rows).sort_values("trajectory")
print("\n=== Implied slopes (percentage points per year) by trajectory ===")
print(slopes.to_string(index=False))

slopes.to_excel(OUT_SLOPES, index=False, engine="openpyxl")
print(f"\nSaved slopes table: {OUT_SLOPES}")

# -----------------------------
# Optional: predicted mean trajectories (fixed effects only)
# -----------------------------
# This is helpful for plotting model-implied trajectories in a later script.
# We'll generate predictions for time=0..9 (2015..2024).
pred_grid = pd.DataFrame({
    "time": np.arange(0, 10, dtype=int),
})

pred_list = []
for g in cats:
    tmp = pred_grid.copy()
    tmp["gavi_trajectory"] = g
    # Need to match the categorical treatment coding
    tmp["gavi_trajectory"] = pd.Categorical(tmp["gavi_trajectory"], categories=cats, ordered=True)
    tmp["pred_mean_cov"] = res.predict(tmp)
    tmp["year"] = tmp["time"] + 2015
    tmp["trajectory"] = g
    pred_list.append(tmp)

pred = pd.concat(pred_list, ignore_index=True)
pred_out = OUT_DIR / "growth_model_predicted_means_by_year.xlsx"
pred.to_excel(pred_out, index=False, engine="openpyxl")
print(f"\nSaved predicted means grid: {pred_out}")

print("\nDONE.")

# --------------------------------------------------
# Clean regression table for publication
# --------------------------------------------------

def star(p):
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.1:
        return "*"
    else:
        return ""

label_map = {
    "Intercept": "Intercept (Never Gavi, 2015)",
    'C(gavi_trajectory, Treatment(reference="Never Gavi (always)"))[T.Classic Gavi (always)]':
        "Classic Gavi (always)",
    'C(gavi_trajectory, Treatment(reference="Never Gavi (always)"))[T.Classic → MIC (graduated)]':
        "Classic → MIC (graduated)",
    'C(gavi_trajectory, Treatment(reference="Never Gavi (always)"))[T.Never → MIC (MICs entry)]':
        "Never → MIC (MICs entry)",
    "time": "Year (since 2015)",
    'time:C(gavi_trajectory, Treatment(reference="Never Gavi (always)"))[T.Classic Gavi (always)]':
        "Year × Classic Gavi",
    'time:C(gavi_trajectory, Treatment(reference="Never Gavi (always)"))[T.Classic → MIC (graduated)]':
        "Year × Classic → MIC",
    'time:C(gavi_trajectory, Treatment(reference="Never Gavi (always)"))[T.Never → MIC (MICs entry)]':
        "Year × Never → MIC",
}

table = (
    fe_table
    .reset_index()
    .rename(columns={"term": "variable"})
    .assign(
        variable=lambda d: d["variable"].map(label_map),
        coef_fmt=lambda d: d["coef"].round(2).astype(str) + d["p_approx"].apply(star),
        se_fmt=lambda d: "(" + d["se"].round(2).astype(str) + ")"
    )
    .loc[lambda d: d["variable"].notna(),
         ["variable", "coef_fmt", "se_fmt"]]
)

print("\n=== Publication-style regression table ===")
print(table.to_string(index=False))

# Save
OUT_REG = OUT_DIR / "growth_model_regression_table.xlsx"
table.to_excel(OUT_REG, index=False, engine="openpyxl")
print(f"\nSaved regression table: {OUT_REG}")
