#!/usr/bin/env python3
# ============================================================
# fig_mean_hpv_first_dose_HIC_vs_nonHIC.py
#
# Purpose (Figure 0 / background motivation):
#   Show mean HPV FIRST-DOSE coverage over time for
#   High-income (HIC) vs Non-high-income (non-HIC) countries.
#
# Why:
#   To demonstrate that convergence is NOT driven by
#   declining HIC coverage, but by rising non-HIC coverage.
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

OUT_FIG = OUT_DIR / "fig_mean_hpv_first_dose_HIC_vs_nonHIC_2015_2024.png"
OUT_DATA = OUT_DIR / "mean_hpv_first_dose_HIC_vs_nonHIC_table.xlsx"

# -----------------------------
# Load
# -----------------------------
df = pd.read_excel(INPUT_XLSX, engine="openpyxl")

# Basic checks
needed = {"country_code", "year", "income_class", "vax_fd_cov"}
missing = needed - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Ensure types
df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
df["income_class"] = df["income_class"].astype("string").str.strip().str.upper()
df["vax_fd_cov"] = pd.to_numeric(df["vax_fd_cov"], errors="coerce")

# Restrict years and non-missing coverage
df = df[df["year"].between(2015, 2024)].copy()
df = df.dropna(subset=["vax_fd_cov", "income_class"]).copy()

# -----------------------------
# Create HIC vs non-HIC indicator
# -----------------------------
df["income_group"] = np.where(
    df["income_class"] == "H",
    "High-income (HIC)",
    "Non-high-income (non-HIC)"
)

# -----------------------------
# Aggregate mean coverage by year and income group
# -----------------------------
mean_cov = (
    df.groupby(["year", "income_group"], as_index=False)
      .agg(
          mean_cov=("vax_fd_cov", "mean"),
          n_countries=("country_code", "nunique"),
          n_obs=("vax_fd_cov", "size"),
      )
      .sort_values(["income_group", "year"])
)

# Save table (for appendix / checking)
mean_cov.to_excel(OUT_DATA, index=False, engine="openpyxl")
print(f"Saved summary table: {OUT_DATA}")

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(8.5, 5.2))

for grp in ["High-income (HIC)", "Non-high-income (non-HIC)"]:
    sub = mean_cov[mean_cov["income_group"] == grp]
    ax.plot(
        sub["year"],
        sub["mean_cov"],
        marker="o",
        linewidth=2,
        label=grp
    )

# Visual context: COVID years
ax.axvspan(2020, 2021, alpha=0.08)

ax.set_title("Mean HPV first-dose coverage over time: HIC vs non-HIC countries")
ax.set_xlabel("Year")
ax.set_ylabel("Mean HPV first-dose coverage (%)")
ax.set_xlim(2015, 2024)

# y-limits with padding
ymin = float(np.nanmin(mean_cov["mean_cov"]))
ymax = float(np.nanmax(mean_cov["mean_cov"]))
pad = 0.05 * (ymax - ymin) if ymax > ymin else 5
ax.set_ylim(ymin - pad, ymax + pad)

ax.grid(True, axis="y", alpha=0.25)
ax.legend(title="Income group", loc="upper left")

note = "\n".join([
    "Notes: vax_fd_cov denotes HPV FIRST-DOSE coverage.",
    "Non-HIC includes low-, lower-middle-, and upper-middle-income countries.",
    "Shaded area marks 2020–2021 (COVID-19 disruption period).",
])
fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=9)

fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(OUT_FIG, dpi=300)
plt.close(fig)

print(f"Saved figure: {OUT_FIG}")

# -----------------------------
# Quick diagnostic print
# -----------------------------
print("\n=== Mean HPV first-dose coverage by income group (2015 vs 2024) ===")
wide = mean_cov.pivot(index="income_group", columns="year", values="mean_cov")
cols = [c for c in [2015, 2024] if c in wide.columns]
print(wide[cols].to_string())
