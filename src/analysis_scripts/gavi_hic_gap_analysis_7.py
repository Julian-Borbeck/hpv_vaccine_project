#!/usr/bin/env python3
# ============================================================
# fig_coverage_by_delivery_model.py
#
# Step 2 (Mechanism – Coverage):
#   Compare HPV first-dose coverage between
#   school-based vs non-school-based delivery models
#   among NON-HIC countries.
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

OUT_TABLE = OUT_DIR / "table_coverage_by_delivery_model_NONHIC.xlsx"
OUT_FIG = OUT_DIR / "fig_coverage_by_delivery_model_NONHIC.png"

# -----------------------------
# Load + prep
# -----------------------------
df = pd.read_excel(INPUT_XLSX, engine="openpyxl")

needed = {
    "country_code", "year", "income_class",
    "vax_fd_cov", "type_prim_deliv_vax"
}
missing = needed - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["income_class"] = df["income_class"].astype("string").str.upper()
df["vax_fd_cov"] = pd.to_numeric(df["vax_fd_cov"], errors="coerce")
df["type_prim_deliv_vax"] = df["type_prim_deliv_vax"].astype("string").str.lower()

# Restrict to non-HIC countries
df = df[df["income_class"] != "H"].copy()

# Keep study years and valid coverage
df = df[df["year"].between(2015, 2024)]
df = df.dropna(subset=["vax_fd_cov", "type_prim_deliv_vax"])

# -----------------------------
# Define delivery indicator
# -----------------------------
df["school_based"] = df["type_prim_deliv_vax"].str.contains(
    "school", case=False, na=False
).astype(int)

df["delivery_model"] = df["school_based"].map({
    1: "School-based",
    0: "Non-school-based"
})

# -----------------------------
# TABLE: Mean coverage by delivery model
# -----------------------------
tab = (
    df.groupby("delivery_model", as_index=False)
      .agg(
          n_country_years=("vax_fd_cov", "size"),
          n_countries=("country_code", "nunique"),
          mean_coverage=("vax_fd_cov", "mean"),
      )
)

tab.to_excel(OUT_TABLE, index=False, engine="openpyxl")
print(f"Saved table: {OUT_TABLE}")

print("\n=== Mean HPV first-dose coverage by delivery model (NON-HIC) ===")
print(tab.to_string(index=False, float_format="%.2f"))

# -----------------------------
# FIGURE: Coverage over time by delivery model
# -----------------------------
ts = (
    df.groupby(["year", "delivery_model"], as_index=False)
      .agg(mean_coverage=("vax_fd_cov", "mean"))
)

fig, ax = plt.subplots(figsize=(7.5, 4.8))

for d in ["School-based", "Non-school-based"]:
    sub = ts[ts["delivery_model"] == d]
    ax.plot(
        sub["year"],
        sub["mean_coverage"],
        marker="o",
        linewidth=2.5,
        label=d
    )

ax.axvspan(2020, 2021, alpha=0.08)  # COVID context
ax.set_title("HPV first-dose coverage by delivery model (non-HIC countries)")
ax.set_xlabel("Year")
ax.set_ylabel("Mean HPV first-dose coverage (%)")
ax.set_xlim(2015, 2024)
ax.set_ylim(0, 100)

ax.grid(True, axis="y", alpha=0.25)
ax.legend(title="Delivery model")

note = (
    "Notes: vax_fd_cov denotes HPV FIRST-DOSE coverage.\n"
    "Sample restricted to non-high-income countries.\n"
    "School-based delivery defined using primary delivery modality."
)
fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=9)

fig.tight_layout(rect=[0, 0.08, 1, 1])
fig.savefig(OUT_FIG, dpi=300)
plt.close(fig)

print(f"Saved figure: {OUT_FIG}")
print("DONE.")
