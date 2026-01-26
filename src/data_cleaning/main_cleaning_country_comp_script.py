# run clean_who_vax
# run wb_income_class
# run gavi and gavi mic country
# run market segment gavi vax price

#!/usr/bin/env python3
import sys
import subprocess
from pathlib import Path

# Get project root (2 levels up from this script)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEANING_SCRIPTS_DIR = PROJECT_ROOT / "src/data_cleaning/cleaning_scripts"

SCRIPTS = [
    CLEANING_SCRIPTS_DIR / "clean_who_vax_cov_first_last_15f.py",
    CLEANING_SCRIPTS_DIR / "wb_income_class_cleaning.py",
    CLEANING_SCRIPTS_DIR / "gavi_and_gavi_mic_country.py",
    CLEANING_SCRIPTS_DIR / "market_segment_gavi_vax_price.py",
    CLEANING_SCRIPTS_DIR / "combine_cleaned_data.py",
    # CLEANING_SCRIPTS_DIR / "coverage.py",
    CLEANING_SCRIPTS_DIR / "coverage_2024.py",
]

def run_one(script_path: str) -> None:
    p = Path(script_path)
    if not p.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    print("\n" + "=" * 90)
    print(f"RUNNING: {p.name}")
    print(f"PATH   : {p}")
    print("=" * 90)

    # Use same Python interpreter running this main script
    result = subprocess.run(
        [sys.executable, str(p)],
        text=True,
        capture_output=True,
    )

    # Print outputs
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        # stderr can include warnings; still print it so you see everything
        print(result.stderr)

    # Hard fail on error
    if result.returncode != 0:
        raise RuntimeError(f"Script failed ({p.name}) with exit code {result.returncode}")

    print(f"DONE: {p.name}")

def main():
    print("Starting full cleaning pipeline...\n")
    for s in SCRIPTS:
        run_one(s)
    print("\nAll cleaning scripts finished successfully.")

if __name__ == "__main__":
    main()
