#!/usr/bin/env python3
# ============================================================
# fig_gap_to_hic_convergence_by_gavi_trajectory.py
#
# EDITED VERSION (key change):
# - High-income countries (income_class == "H") are used ONLY to build the
#   yearly benchmark (HIC_mean_t).
# - HIC rows are EXCLUDED from the trajectory comparison groups/lines,
#   so the "Never Gavi" line does not mechanically include HICs.
#
# Outcome:
#   Plot mean gap to HIC mean over time for NON-HIC countries only.
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

OUT_FIG = OUT_DIR / "fig_gap_to_HIC_mean_by_gavi_trajectory_NONHIC_2015_2024.png"
OUT_DATA = OUT_DIR / "gap_to_HIC_table_by_year_and_trajectory_NONHIC.xlsx"

# -----------------------------
# Load
# -----------------------------
df = pd.read_excel(INPUT_XLSX, engine="openpyxl")

# Basic checks
needed = {"country_code", "year", "income_class", "vax_fd_cov", "gavi_trajectory"}
missing = needed - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Ensure types
df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
df["income_class"] = df["income_class"].astype("string").str.strip().str.upper()
df["vax_fd_cov"] = pd.to_numeric(df["vax_fd_cov"], errors="coerce")

# Keep years of interest and non-missing key fields
df = df[df["year"].between(2015, 2024)].copy()
df = df.dropna(subset=["vax_fd_cov", "income_class", "gavi_trajectory"]).copy()

# -----------------------------
# 1) Compute HIC mean coverage by year (benchmark)
#    IMPORTANT: benchmark uses ONLY income_class == "H"
# -----------------------------
hic_mean_by_year = (
    df.loc[df["income_class"] == "H"]
    .groupby("year")["vax_fd_cov"]
    .mean()
    .rename("hic_mean_vax_fd_cov")
    .reset_index()
)

if hic_mean_by_year.empty:
    raise ValueError("No HIC observations found (income_class == 'H'). Cannot compute benchmark.")

# Merge benchmark to all rows (so we can compute gaps)
df = df.merge(hic_mean_by_year, on="year", how="left")

# -----------------------------
# 2) Compute gap to HIC mean
# -----------------------------
df["gap_to_hic_mean"] = df["hic_mean_vax_fd_cov"] - df["vax_fd_cov"]

# -----------------------------
# 2b) EXCLUDE HIC rows from the comparison groups/lines
# -----------------------------
df_nonhic = df[df["income_class"] != "H"].copy()

# Sanity check
print("\n=== Non-HIC analysis sample (after excluding income_class == 'H') ===")
print("Rows (country-years):", df_nonhic.shape[0])
print("Unique countries:", df_nonhic["country_code"].nunique())
print("income_class distribution:")
print(df_nonhic["income_class"].value_counts(dropna=False).to_string())

# -----------------------------
# 3) Aggregate: mean gap by year and trajectory (NON-HIC only)
# -----------------------------
gap_summary = (
    df_nonhic.groupby(["year", "gavi_trajectory"], as_index=False)
      .agg(
          mean_gap=("gap_to_hic_mean", "mean"),
          n_obs=("gap_to_hic_mean", "size"),
          n_countries=("country_code", "nunique"),
          mean_cov=("vax_fd_cov", "mean"),
      )
      .sort_values(["gavi_trajectory", "year"])
)

# Save the table for your appendix / checking
gap_summary.to_excel(OUT_DATA, index=False, engine="openpyxl")
print(f"\nSaved summary table: {OUT_DATA}")

# -----------------------------
# 4) Plot: mean gap trajectories over time (NON-HIC only)
# -----------------------------
trajectory_order = [
    "Classic Gavi (always)",
    "Classic → MIC (graduated)",
    "Never → MIC (MICs entry)",
    "Never Gavi (always)",
]

present = [t for t in trajectory_order if t in gap_summary["gavi_trajectory"].unique().tolist()]
if not present:
    raise ValueError("No recognized gavi_trajectory categories found in NON-HIC data.")

fig, ax = plt.subplots(figsize=(9, 5.2))

for traj in present:
    sub = gap_summary[gap_summary["gavi_trajectory"] == traj].sort_values("year")
    ax.plot(sub["year"], sub["mean_gap"], marker="o", linewidth=2, label=traj)

# Reference line: 0 gap means equal to HIC mean
ax.axhline(0, linewidth=1)

# Mark the 2022 regime-change period (informative, not causal)
ax.axvline(2022, linestyle="--", linewidth=1)

ax.set_title("HPV Vaccine 1st dose coverage gap to HIC mean")
ax.set_xlabel("Year")
ax.set_ylabel("Mean gap to HIC mean (percentage points)\n(HIC mean minus country coverage)")
ax.set_xlim(2015, 2024)

# y-limits with some headroom
ymin = float(np.nanmin(gap_summary["mean_gap"]))
ymax = float(np.nanmax(gap_summary["mean_gap"]))
pad = 0.05 * (ymax - ymin) if ymax > ymin else 5
ax.set_ylim(ymin - pad, ymax + pad)

ax.grid(True, axis="y", alpha=0.25)
ax.legend(title="Trajectory", bbox_to_anchor=(1.02, 1), loc="upper left")

note = "\n".join([
    "Notes: vax_fd_cov is HPV FIRST-DOSE coverage.",
    "High-income (H) countries are used ONLY to construct the yearly benchmark and are excluded from the plotted groups.",
    "Gap is computed relative to the mean coverage among high-income (H) countries in each year.",
    "Dashed vertical line marks 2022 (MICs approach rollout / regime transitions).",
])
fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=9)

fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(OUT_FIG, dpi=300)
plt.close(fig)

print(f"\nSaved figure: {OUT_FIG}")

# -----------------------------
# 5) Quick prints for interpretation
# -----------------------------
print("\n=== HIC benchmark (mean first-dose coverage) by year ===")
print(hic_mean_by_year.to_string(index=False))

print("\n=== Mean gap in 2015 vs 2024 by trajectory (NON-HIC only) ===")
wide = gap_summary.pivot(index="gavi_trajectory", columns="year", values="mean_gap")
cols = [c for c in [2015, 2024] if c in wide.columns]
print(wide[cols].sort_index().to_string())
