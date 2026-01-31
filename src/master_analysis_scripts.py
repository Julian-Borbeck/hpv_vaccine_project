# ============================================================
# master_analysis_country.py
# Master runner for analysis scripts
# ============================================================

import sys
import subprocess
from pathlib import Path

# ------------------------------------------------------------
# Ordered analysis pipeline
# (descriptives -> regimes/trajectory -> hic gap series -> extras)
# ------------------------------------------------------------
SCRIPTS = [

    # 3) HIC gap analyses (iterative versions)
    r"/Users/khaira_abdillah/Documents/dl_pro_country_comp/scripts/analysis_scripts/gavi_hic_gap_analysis_1.py",
    r"/Users/khaira_abdillah/Documents/dl_pro_country_comp/scripts/analysis_scripts/gavi_hic_gap_analysis_2.py",
    r"/Users/khaira_abdillah/Documents/dl_pro_country_comp/scripts/analysis_scripts/gavi_hic_gap_analysis_3.py", #growth model regression
    r"/Users/khaira_abdillah/Documents/dl_pro_country_comp/scripts/analysis_scripts/gavi_hic_gap_analysis_4.py", #growth model predicted fig with raw
    r"/Users/khaira_abdillah/Documents/dl_pro_country_comp/scripts/analysis_scripts/gavi_hic_gap_analysis_5.py", #growth model predicted fig only
    r"/Users/khaira_abdillah/Documents/dl_pro_country_comp/scripts/analysis_scripts/gavi_hic_gap_analysis_6.py", #school-based delivery
    r"/Users/khaira_abdillah/Documents/dl_pro_country_comp/scripts/analysis_scripts/gavi_hic_gap_analysis_7.py", #coverage by delivery model NONHIC
    
]

# ------------------------------------------------------------
# Runner
# ------------------------------------------------------------
def run_one(script_path: str) -> None:
    p = Path(script_path)
    if not p.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    print("\n" + "=" * 90)
    print(f"RUNNING: {p.name}")
    print(f"PATH   : {p}")
    print("=" * 90)

    result = subprocess.run(
        [sys.executable, str(p)],
        text=True,
        capture_output=True,
    )

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"❌ Script failed ({p.name}) with exit code {result.returncode}")

    print(f"✅ DONE: {p.name}")

def main():
    print("Starting full analysis pipeline...\n")
    for s in SCRIPTS:
        run_one(s)
    print("\n🎉 All analysis scripts finished successfully.")

if __name__ == "__main__":
    main()
