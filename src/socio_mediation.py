"""
Mediation Analysis Module for HPV Vaccine Coverage Study

This module provides functions for:
1. Traditional statistical tests (Chi-square, ANOVA, t-test)
2. Regression-based mediation analysis using Pingouin
3. Result extraction and summarization
"""

import pandas as pd
import numpy as np
from scipy import stats
import pingouin as pg
from pathlib import Path

# =============================================================================
# Merge data and save functions
# =============================================================================


def merge_delivery_dosing(dosing_path, delivery_path):
    """Merge delivery strategy with dosing schedule data."""
    dosing_df = pd.read_csv(dosing_path)      
    delivery_df = pd.read_csv(delivery_path)  

    # Merge on country code
    df_merged = delivery_df.merge(
        dosing_df[['ISO', 'Gavi Status']],
        left_on='ISO_3_CODE',
        right_on='ISO',
        how='inner'
    )
    
    # Clean up
    df_merged = df_merged.drop('ISO', axis=1)
    df_merged['HPV_YEAR_INTRODUCTION'] = df_merged['HPV_YEAR_INTRODUCTION'].astype('Int64')
    
    return df_merged

def save_merged_data(df_merged, output_path):
    """Save the merged DataFrame to cleaned folder."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(output_path, index=False)

# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def load_and_prepare_data(dosing_path: str, delivery_path: str) -> pd.DataFrame:
    """
    Load and merge dosing and delivery data, prepare for mediation analysis.

    Parameters
    ----------
    dosing_path : str
        Path to current_dosing.csv
    delivery_path : str
        Path to delivery_strategy.csv

    Returns
    -------
    pd.DataFrame
        Merged and encoded dataframe ready for analysis
    """
    # Load data
    dosing_df = pd.read_csv(dosing_path)
    delivery_df = pd.read_csv(delivery_path)

    # Merge
    df = delivery_df.merge(
        dosing_df[['ISO', 'Gavi Status']],
        left_on='ISO_3_CODE', right_on='ISO', how='inner'
    ).drop('ISO', axis=1)

    # Filter complete cases
    df_analysis = df[
        df['HPV1_COVERAGELASTYEAR'].notna() &
        df['HPV_PRIM_DELIV_STRATEGY'].notna() &
        df['HPV_INT_DOSES'].notna() &
        df['Gavi Status'].notna()
    ].copy()

    # Encode variables
    df_analysis['X'] = (df_analysis['Gavi Status'] == 'Gavi').astype(int)
    df_analysis['Y'] = df_analysis['HPV1_COVERAGELASTYEAR']

    # Numeric encoding for mediators
    delivery_map = {
        'School-based': 0,
        'Facility-based': 1,
        'mixed': 2,
        'Varies by region/province': 3
    }
    dose_map = {
        '1 dose': 1,
        '2 doses (6 months)': 2,
        '2 doses (12 months)': 3
    }

    df_analysis['M1'] = df_analysis['HPV_PRIM_DELIV_STRATEGY'].map(delivery_map)
    df_analysis['M2'] = df_analysis['HPV_INT_DOSES'].map(dose_map)

    return df_analysis


# =============================================================================
# PART 1: TRADITIONAL STATISTICAL TESTS
# =============================================================================

def traditional_mediation_test(df: pd.DataFrame, mediator_col: str) -> dict:
    """
    Perform traditional mediation tests (Chi-square, ANOVA, t-test).

    Parameters
    ----------
    df : pd.DataFrame
        Prepared dataframe with X, Y columns
    mediator_col : str
        Column name for the categorical mediator variable

    Returns
    -------
    dict
        Dictionary containing all test statistics and p-values
    """
    results = {}

    # PATH A: Chi-square test (X → M)
    contingency = pd.crosstab(df['X'], df[mediator_col])
    chi2, p_a, dof, expected = stats.chi2_contingency(contingency)
    results['path_a'] = {
        'test': 'Chi-square',
        'statistic': chi2,
        'p_value': p_a,
        'df': dof,
        'contingency_table': contingency
    }

    # PATH B: ANOVA (M → Y)
    mediator_values = df[mediator_col].dropna().unique()
    groups = [df[df[mediator_col] == val]['Y'].values for val in mediator_values]
    f_stat, p_b = stats.f_oneway(*groups)

    mediator_means = df.groupby(mediator_col)['Y'].agg(['mean', 'count', 'std']).round(4)
    results['path_b'] = {
        'test': 'ANOVA',
        'statistic': f_stat,
        'p_value': p_b,
        'group_means': mediator_means
    }

    # PATH C: t-test (X → Y)
    gavi = df[df['X'] == 1]['Y']
    non_gavi = df[df['X'] == 0]['Y']
    t_stat, p_c = stats.ttest_ind(gavi, non_gavi)

    results['path_c'] = {
        'test': 't-test',
        'statistic': t_stat,
        'p_value': p_c,
        'gavi_mean': gavi.mean(),
        'non_gavi_mean': non_gavi.mean(),
        'mean_diff': gavi.mean() - non_gavi.mean(),
        'n_gavi': len(gavi),
        'n_non_gavi': len(non_gavi)
    }

    return results


def print_traditional_results(results: dict, model_name: str) -> None:
    """Print formatted traditional test results."""
    print("=" * 70)
    print(f"TRADITIONAL TESTS: {model_name}")
    print("=" * 70)

    # Path A
    a = results['path_a']
    sig_a = "Significant" if a['p_value'] < 0.05 else "Not significant"
    print(f"\nPath A (X → M): {a['test']}")
    print(f"  χ² = {a['statistic']:.4f}, df = {a['df']}, p = {a['p_value']:.4f}")
    print(f"  Result: {sig_a}")

    # Path B
    b = results['path_b']
    sig_b = "Significant" if b['p_value'] < 0.05 else "Not significant"
    print(f"\nPath B (M → Y): {b['test']}")
    print(f"  F = {b['statistic']:.4f}, p = {b['p_value']:.4f}")
    print(f"  Result: {sig_b}")

    # Path C
    c = results['path_c']
    sig_c = "Significant" if c['p_value'] < 0.05 else "Not significant"
    print(f"\nPath C (X → Y): {c['test']}")
    print(f"  Gavi mean: {c['gavi_mean']:.3f} (n={c['n_gavi']})")
    print(f"  Non-Gavi mean: {c['non_gavi_mean']:.3f} (n={c['n_non_gavi']})")
    print(f"  Difference: {c['mean_diff']:.3f}")
    print(f"  t = {c['statistic']:.4f}, p = {c['p_value']:.4f}")
    print(f"  Result: {sig_c}")


# =============================================================================
# PART 2: REGRESSION-BASED MEDIATION (PINGOUIN)
# =============================================================================

def regression_mediation(df: pd.DataFrame, x: str, m: str, y: str,
                         n_boot: int = 5000, seed: int = 42) -> dict:
    """
    Perform regression-based mediation analysis using Pingouin.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with variables
    x : str
        Column name for independent variable
    m : str
        Column name for mediator variable
    y : str
        Column name for dependent variable
    n_boot : int
        Number of bootstrap iterations
    seed : int
        Random seed for reproducibility

    Returns
    -------
    dict
        Dictionary containing regression results and mediation analysis
    """
    results = {}

    # Path A: X → M
    path_a = pg.linear_regression(df[[x]], df[m])
    results['path_a_regression'] = path_a

    # Path B + C': X + M → Y
    path_bc = pg.linear_regression(df[[x, m]], df[y])
    results['path_bc_regression'] = path_bc

    # Full mediation analysis with bootstrap
    mediation = pg.mediation_analysis(
        data=df, x=x, m=m, y=y,
        n_boot=n_boot, seed=seed
    )
    results['mediation'] = mediation

    # Extract key coefficients
    results['coefficients'] = extract_mediation_coefficients(mediation)

    return results


def extract_mediation_coefficients(med_df: pd.DataFrame) -> dict:
    """
    Extract coefficients from pingouin mediation_analysis output.

    Parameters
    ----------
    med_df : pd.DataFrame
        Output from pg.mediation_analysis()

    Returns
    -------
    dict
        Dictionary with all path coefficients
    """
    coefs = {}

    # Map pingouin path names to our naming convention
    path_map = {
        'a': None,  # Will be extracted from pattern matching
        'b': None,
        'total': 'Total',
        'direct': 'Direct',
        'indirect': 'Indirect'
    }

    for _, row in med_df.iterrows():
        path = row['path']

        if ' ~ ' in path:  # Path a or b (e.g., "M1 ~ X" or "Y ~ M1")
            if path.startswith('Y'):
                key = 'b'
            else:
                key = 'a'
        elif path == 'Total':
            key = 'total'
        elif path == 'Direct':
            key = 'direct'
        elif path == 'Indirect':
            key = 'indirect'
        else:
            continue

        coefs[key] = {
            'coef': row['coef'],
            'se': row['se'] if pd.notna(row['se']) else None,
            'pval': row['pval'] if pd.notna(row['pval']) else None,
            'ci_lower': row['CI[2.5%]'],
            'ci_upper': row['CI[97.5%]'],
            'sig': row['sig'] if 'sig' in row else None
        }

    return coefs


def print_regression_results(results: dict, model_name: str) -> None:
    """Print formatted regression mediation results."""
    print("=" * 70)
    print(f"REGRESSION MEDIATION: {model_name}")
    print("=" * 70)

    coefs = results['coefficients']

    print("\nPath Coefficients:")
    print("-" * 50)

    for path, data in coefs.items():
        sig = "*" if data['pval'] and data['pval'] < 0.05 else ""
        pval_str = f"p = {data['pval']:.4f}" if data['pval'] else "p = N/A"
        ci_str = f"[{data['ci_lower']:.4f}, {data['ci_upper']:.4f}]"
        print(f"  {path:10s}: β = {data['coef']:.4f}, {pval_str}, 95% CI {ci_str} {sig}")

    # Check mediation significance
    indirect = coefs.get('indirect', {})
    ci_l = indirect.get('ci_lower', 0)
    ci_u = indirect.get('ci_upper', 0)

    if (ci_l > 0) or (ci_u < 0):
        print("\n  → Significant mediation (CI excludes zero)")
    else:
        print("\n  → No significant mediation (CI includes zero)")


# =============================================================================
# SUMMARY FUNCTIONS
# =============================================================================

def create_summary_table(results_delivery: dict, results_dosing: dict) -> pd.DataFrame:
    """
    Create summary table comparing both mediation models.

    Parameters
    ----------
    results_delivery : dict
        Results from regression_mediation for delivery strategy
    results_dosing : dict
        Results from regression_mediation for dosing schedule

    Returns
    -------
    pd.DataFrame
        Summary comparison table
    """
    def format_coef(data):
        if data['pval']:
            return f"{data['coef']:.4f} (p={data['pval']:.3f})"
        return f"{data['coef']:.4f}"

    def format_ci(data):
        return f"[{data['ci_lower']:.4f}, {data['ci_upper']:.4f}]"

    def is_significant(data):
        ci_l = data.get('ci_lower', 0)
        ci_u = data.get('ci_upper', 0)
        return 'Yes' if (ci_l > 0) or (ci_u < 0) else 'No'

    c1 = results_delivery['coefficients']
    c2 = results_dosing['coefficients']

    summary = pd.DataFrame({
        'Mediator': ['Delivery Strategy', 'Dosing Schedule'],
        'Path a (X→M)': [format_coef(c1['a']), format_coef(c2['a'])],
        'Path b (M→Y)': [format_coef(c1['b']), format_coef(c2['b'])],
        'Direct (c\')': [format_coef(c1['direct']), format_coef(c2['direct'])],
        'Indirect (a×b)': [f"{c1['indirect']['coef']:.4f}", f"{c2['indirect']['coef']:.4f}"],
        '95% CI': [format_ci(c1['indirect']), format_ci(c2['indirect'])],
        'Mediation': [is_significant(c1['indirect']), is_significant(c2['indirect'])]
    })

    return summary


def run_full_analysis(df: pd.DataFrame, n_boot: int = 5000, seed: int = 42) -> dict:
    """
    Run complete mediation analysis for both models.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared dataframe
    n_boot : int
        Number of bootstrap iterations
    seed : int
        Random seed

    Returns
    -------
    dict
        All results for both models
    """
    results = {}

    # Model 1: Delivery Strategy
    print("\n" + "=" * 70)
    print("MODEL 1: Gavi → Delivery Strategy → Coverage")
    print("=" * 70)

    results['delivery_traditional'] = traditional_mediation_test(
        df, 'HPV_PRIM_DELIV_STRATEGY'
    )
    results['delivery_regression'] = regression_mediation(
        df, 'X', 'M1', 'Y', n_boot=n_boot, seed=seed
    )

    # Model 2: Dosing Schedule
    print("\n" + "=" * 70)
    print("MODEL 2: Gavi → Dosing Schedule → Coverage")
    print("=" * 70)

    results['dosing_traditional'] = traditional_mediation_test(
        df, 'HPV_INT_DOSES'
    )
    results['dosing_regression'] = regression_mediation(
        df, 'X', 'M2', 'Y', n_boot=n_boot, seed=seed
    )

    # Summary table
    results['summary'] = create_summary_table(
        results['delivery_regression'],
        results['dosing_regression']
    )

    return results


# =============================================================================
# MAIN EXECUTION (for testing)
# =============================================================================

if __name__ == "__main__":
    # Get project root
    project_root = Path(__file__).parent.parent

    dosing_path = project_root / 'dat' / 'Socio_Econ' / 'raw' / 'current_dosing.csv'
    delivery_path = project_root / 'dat' / 'Socio_Econ' / 'raw' / 'delivery_strategy.csv'

    # merge delivery + dosing and save cleaned file
    output_path = project_root / 'dat' / 'Socio_Econ' / 'cleaned' / 'merged_delivery_dosing.csv'
    merged_df = merge_delivery_dosing(dosing_path, delivery_path)
    save_merged_data(merged_df, output_path)

    # Load data
    df = load_and_prepare_data(
        dosing_path=dosing_path,
        delivery_path=delivery_path
    )

    print(f"Loaded {len(df)} countries with complete data")
    print(f"Gavi: {df['X'].sum()}, Non-Gavi: {(df['X']==0).sum()}")

    # Run analysis
    results = run_full_analysis(df, n_boot=5000, seed=42)

    # Print results
    print_traditional_results(results['delivery_traditional'], "Delivery Strategy")
    print_regression_results(results['delivery_regression'], "Delivery Strategy")

    print_traditional_results(results['dosing_traditional'], "Dosing Schedule")
    print_regression_results(results['dosing_regression'], "Dosing Schedule")

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(results['summary'].to_string(index=False))
