# main.py - HPV Vaccine Project Analysis Pipeline
from pathlib import Path

# Socio-Econ imports

from socio_mediation import (
    merge_delivery_dosing, 
    save_merged_data,
    load_and_prepare_data,
    run_full_analysis,
    print_traditional_results,
    print_regression_results
)
from socio_section1_income import run_section1_analysis
from socio_section2_gavi import run_section2_gavi_analysis
from socio_section2_price import run_section2_price_analysis
from socio_section3_pca import run_section3_pca_analysis


def main():
    """Run complete HPV vaccination analysis pipeline."""

    # Get project root directory (parent of src/)
    project_root = Path(__file__).parent.parent

    # Paths
    socio_data = project_root / 'dat' / 'Socio_Econ' / 'cleaned' / 'dl_project_section_1.xlsx'
    socio_raw = project_root / 'dat' / 'Socio_Econ' / 'raw'
    output_dir = project_root / 'doc' / 'fig' / 'Socio_Econ'

    # ==========================================================================
    # SOCIO-ECONOMIC ANALYSIS
    # ==========================================================================

    # Section 1: Income Class Analysis
    run_section1_analysis(str(socio_data), str(output_dir))

    # Section 2: Gavi Eligibility Analysis
    run_section2_gavi_analysis(str(socio_data), str(output_dir))

    # Section 2: Vaccine Price Analysis
    run_section2_price_analysis(str(socio_data), str(output_dir))

    # Section 3: PCA Analysis
    run_section3_pca_analysis(str(socio_data), str(output_dir))

    # Mediation Analysis
    print("\n" + "=" * 70)
    print("MEDIATION ANALYSIS")
    print("=" * 70)

    df = load_and_prepare_data(
        dosing_path=socio_raw / 'current_dosing.csv',
        delivery_path=socio_raw / 'delivery_strategy.csv'
    )
    print(f"\nLoaded {len(df)} countries with complete data")
    print(f"Gavi: {df['X'].sum()}, Non-Gavi: {(df['X']==0).sum()}")

    results = run_full_analysis(df, n_boot=5000, seed=42)

    print("\n" + "=" * 70)
    print("TRADITIONAL TESTS RESULTS")
    print("=" * 70)
    print_traditional_results(results['delivery_traditional'], "Delivery Strategy")
    print_traditional_results(results['dosing_traditional'], "Dosing Schedule")

    print("\n" + "=" * 70)
    print("REGRESSION MEDIATION RESULTS")
    print("=" * 70)
    print_regression_results(results['delivery_regression'], "Delivery Strategy")
    print_regression_results(results['dosing_regression'], "Dosing Schedule")

    print("\n" + "=" * 70)
    print("FINAL MEDIATION SUMMARY")
    print("=" * 70)
    print(results['summary'].to_string(index=False))

    # ==========================================================================
    # ANALYSIS (future)
    # ==========================================================================


    print("\n" + "=" * 70)
    print("Analysis pipeline complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
