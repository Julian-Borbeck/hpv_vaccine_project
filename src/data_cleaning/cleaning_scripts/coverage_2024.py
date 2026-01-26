"""
Build coverage_2024.csv with first/last dose coverage and delivery strategy.

Steps:
1) Filter coverage_cleaned.csv to YEAR == 2024 and rename COVERAGE -> FIRST_COVERAGE.
2) Merge delivery_strategy.csv (ISO_3_CODE) to add HPV_PRIM_DELIV_STRATEGY.
3) Merge last_dose_coverage.xlsx (YEAR == 2024) and rename COVERAGE -> LAST_COVERAGE.
4) Merge gavi_country_2024 sheet from dl_project_section_1.xlsx to add gavi_2024.
5) Merge current_dosing.csv (ISO) to add Gavi Status.
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

IN_COVERAGE = PROJECT_ROOT / "dat/Socio_Econ/02_cleaned_data/coverage_cleaned.csv"
DELIVERY_PATH = PROJECT_ROOT / "dat/Socio_Econ/00_raw_data/delivery_strategy.csv"
LAST_DOSE_PATH = PROJECT_ROOT / "dat/Socio_Econ/00_raw_data/last_dose_coverage.xlsx"
DOSING_PATH = PROJECT_ROOT / "dat/Socio_Econ/00_raw_data/current_dosing.csv"

OUT_COVERAGE = PROJECT_ROOT / "dat/Socio_Econ/02_cleaned_data/coverage_2024.csv"
GAVI_BOOK = PROJECT_ROOT / "dat/Socio_Econ/02_cleaned_data/dl_project_section_1.xlsx"
GAVI_SHEET = "gavi_country_2024"


def main() -> None:
    df = pd.read_csv(IN_COVERAGE)
    df = df[df["YEAR"] == 2024].copy()
    df = df.rename(columns={"COVERAGE": "FIRST_COVERAGE"})

    delivery = pd.read_csv(DELIVERY_PATH)
    delivery = delivery[["ISO_3_CODE", "HPV_PRIM_DELIV_STRATEGY"]].copy()
    df = df.merge(delivery, left_on="CODE", right_on="ISO_3_CODE", how="left")
    df = df.drop(columns=["ISO_3_CODE"])

    last = pd.read_excel(LAST_DOSE_PATH, engine="openpyxl")
    last = last[last["YEAR"] == 2024].copy()
    last = last[["CODE", "COVERAGE"]].rename(columns={"COVERAGE": "LAST_COVERAGE"})

    df = df.merge(last, on="CODE", how="left")

    gavi = pd.read_excel(GAVI_BOOK, sheet_name=GAVI_SHEET, engine="openpyxl")
    gavi = gavi[["country_code", "gavi_2024"]].copy()
    df = df.merge(gavi, left_on="CODE", right_on="country_code", how="left")
    df = df.drop(columns=["country_code"])

    dosing = pd.read_csv(DOSING_PATH)
    dosing = dosing[["ISO", "Gavi Status"]].copy()
    df = df.merge(dosing, left_on="CODE", right_on="ISO", how="left")
    df = df.drop(columns=["ISO"])

    OUT_COVERAGE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_COVERAGE, index=False)
    print("Saved:", OUT_COVERAGE)


if __name__ == "__main__":
    main()
