#!/usr/bin/env python3
# ============================================================
# tab_school_based_delivery_by_gavi_trajectory.py
#
# Step 1 (Mechanism – Delivery):
#   Are Gavi-supported countries more likely to adopt
#   school-based HPV vaccination delivery?
#
# Output:
#   - Table: share of country-years with school-based delivery
#     by Gavi policy trajectory
# ============================================================

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Paths
# -----------------------------
INPUT_XLSX = Path(
    r"/Users/khaira_abdillah/Documents/dl_pro_country_comp/02_cleaned_data/"
    r"dataset_country_analysis_with_gavi_trajectory.xlsx"
)

OUT_DIR = Path(
    r"/Users/khaira_abdillah/Documents/dl_pro_country_comp/03_outputs/mechanism_delivery"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_TABLE = OUT_DIR / "table_school_based_delivery_by_gavi_trajectory.xlsx"
OUT_FIG = OUT_DIR / "fig_school_based_delivery_by_gavi_trajectory.png"

# -----------------------------
# Load + prep
# -----------------------------
df = pd.read_excel(INPUT_XLSX, engine="openpyxl")

needed = {"country_code", "year", "gavi_trajectory", "type_prim_deliv_vax"}
missing = needed - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["gavi_trajectory"] = df["gavi_trajectory"].astype("string").str.strip()
df["type_prim_deliv_vax"] = df["type_prim_deliv_vax"].astype("string").str.lower()

# Keep study period
df = df[df["year"].between(2015, 2024)].copy()

# -----------------------------
# Define school-based indicator
# -----------------------------
# Adjust keywords if your coding differs
school_keywords = ["school"]

df["school_based"] = df["type_prim_deliv_vax"].str.contains(
    "|".join(school_keywords),
    case=False,
    na=False
).astype(int)

# Drop rows with no delivery information
df = df.dropna(subset=["type_prim_deliv_vax", "gavi_trajectory"])

# -----------------------------
# Aggregate: share of country-years
# -----------------------------
tab = (
    df.groupby("gavi_trajectory", as_index=False)
      .agg(
          n_country_years=("school_based", "size"),
          n_school_based=("school_based", "sum"),
          share_school_based=("school_based", "mean"),
      )
      .sort_values("share_school_based", ascending=False)
)

# Convert share to percent
tab["share_school_based_pct"] = 100 * tab["share_school_based"]

# Save table
tab.to_excel(OUT_TABLE, index=False, engine="openpyxl")
print(f"Saved table: {OUT_TABLE}")

print("\n=== Share of country-years with school-based delivery ===")
print(
    tab[[
        "gavi_trajectory",
        "n_country_years",
        "n_school_based",
        "share_school_based_pct"
    ]].to_string(index=False, float_format="%.1f")
)

# -----------------------------
# Optional figure (bar chart)
# -----------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))

ax.barh(
    tab["gavi_trajectory"],
    tab["share_school_based_pct"]
)

ax.set_xlabel("Share of country-years with school-based delivery (%)")
ax.set_title("School-based HPV vaccination delivery by Gavi trajectory")

ax.set_xlim(0, 100)
ax.grid(axis="x", alpha=0.3)

fig.tight_layout()
fig.savefig(OUT_FIG, dpi=300)
plt.close(fig)

print(f"Saved figure: {OUT_FIG}")
print("DONE.")
