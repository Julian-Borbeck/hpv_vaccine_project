"""
Two-panel mediation diagrams for socio-economic analysis:
1) Traditional stats (chi-square / ANOVA / t-test)
2) Beta coefficients (regression mediation)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from socio_mediation import (
    load_and_prepare_data,
    run_full_analysis,
    traditional_mediation_test,
)


def fmt(coef: float, pval: float) -> tuple:
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
    color = "#2E7D32" if pval < 0.05 else "#757575"
    weight = "bold" if pval < 0.05 else "normal"
    lw = 2.5 if pval < 0.05 else 1.5
    return f"β = {coef:.3f}{sig}", color, weight, lw


def draw_mediation_beta(ax, mediator: str, data: dict) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    X, M, Y = (1.5, 3), (5, 6.5), (8.5, 3)

    for pos, label, color in [
        (X, "Gavi Status", "#E8F4FD"),
        (M, mediator, "#FFF9E6"),
        (Y, "HPV Coverage", "#E8F8E8"),
    ]:
        ax.add_patch(
            plt.Rectangle(
                (pos[0] - 1.2, pos[1] - 0.6),
                2.4,
                1.2,
                facecolor=color,
                edgecolor="#333",
                linewidth=1.5,
                zorder=2,
            )
        )
        ax.text(
            pos[0],
            pos[1],
            label,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            zorder=3,
        )

    a_txt, a_col, a_wt, a_lw = fmt(data["a_coef"], data["a_p"])
    ax.annotate(
        "",
        xy=(M[0] - 1, M[1] - 0.6),
        xytext=(X[0] + 0.8, X[1] + 0.6),
        arrowprops=dict(arrowstyle="-|>", color=a_col, lw=a_lw),
    )
    ax.text(
        3.0,
        5.8,
        f"Path a\n{a_txt}",
        fontsize=10,
        color=a_col,
        fontweight=a_wt,
        ha="center",
        va="center",
        linespacing=1.3,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8),
    )

    b_txt, b_col, b_wt, b_lw = fmt(data["b_coef"], data["b_p"])
    ax.annotate(
        "",
        xy=(Y[0] - 0.8, Y[1] + 0.6),
        xytext=(M[0] + 1, M[1] - 0.6),
        arrowprops=dict(arrowstyle="-|>", color=b_col, lw=b_lw),
    )
    ax.text(
        7.0,
        5.8,
        f"Path b\n{b_txt}",
        fontsize=10,
        color=b_col,
        fontweight=b_wt,
        ha="center",
        va="center",
        linespacing=1.3,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8),
    )

    c_txt, c_col, c_wt, c_lw = fmt(data["c_coef"], data["c_p"])
    ax.annotate(
        "",
        xy=(Y[0] - 1.2, Y[1]),
        xytext=(X[0] + 1.2, X[1]),
        arrowprops=dict(arrowstyle="-|>", color=c_col, lw=c_lw, linestyle="--"),
    )
    ax.text(
        5,
        1.9,
        f"Path c'\n{c_txt}",
        fontsize=10,
        color=c_col,
        fontweight=c_wt,
        ha="center",
        va="center",
        linespacing=1.3,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8),
    )

    ind_p = data["indirect_p"]
    sig_ind = "***" if ind_p < 0.001 else "**" if ind_p < 0.01 else "*" if ind_p < 0.05 else ""
    color_ind = "#2E7D32" if ind_p < 0.05 else "#757575"
    ax.text(
        5,
        7.6,
        f"Indirect: a×b = {data['indirect']:.4f}{sig_ind}",
        ha="center",
        fontsize=11,
        color=color_ind,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8),
    )


def draw_mediation_traditional(ax, mediator: str, stats: dict) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    X, M, Y = (1.5, 3), (5, 6.5), (8.5, 3)

    for pos, label, color in [
        (X, "Gavi Status", "#E8F4FD"),
        (M, mediator, "#FFF9E6"),
        (Y, "HPV Coverage", "#E8F8E8"),
    ]:
        ax.add_patch(
            plt.Rectangle(
                (pos[0] - 1.2, pos[1] - 0.6),
                2.4,
                1.2,
                facecolor=color,
                edgecolor="#333",
                linewidth=1.5,
                zorder=2,
            )
        )
        ax.text(
            pos[0],
            pos[1],
            label,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            zorder=3,
        )

    def fmt_stat(stat_val: float, pval: float, stat_type: str) -> tuple:
        color = "#2E7D32" if pval < 0.05 else "#757575"
        weight = "bold" if pval < 0.05 else "normal"
        lw = 2.0 if pval < 0.05 else 1.0
        if stat_type == "chi2":
            label = f"χ² = {stat_val:.2f}"
        elif stat_type == "F":
            label = f"F = {stat_val:.2f}"
        else:
            label = f"t = {stat_val:.2f}"
        return label, color, weight, lw

    a_stat = stats["path_a"]["statistic"]
    a_p = stats["path_a"]["p_value"]
    b_stat = stats["path_b"]["statistic"]
    b_p = stats["path_b"]["p_value"]
    c_stat = stats["path_c"]["statistic"]
    c_p = stats["path_c"]["p_value"]

    a_txt, a_col, a_wt, a_lw = fmt_stat(a_stat, a_p, "chi2")
    ax.annotate(
        "",
        xy=(M[0] - 1, M[1] - 0.6),
        xytext=(X[0] + 0.8, X[1] + 0.6),
        arrowprops=dict(arrowstyle="-|>", color=a_col, lw=a_lw),
    )
    ax.text(
        3.0,
        5.9,
        f"Path a\n{a_txt}\np = {a_p:.3f}",
        fontsize=10,
        color=a_col,
        fontweight=a_wt,
        ha="center",
        va="center",
        linespacing=1.3,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8),
    )

    b_txt, b_col, b_wt, b_lw = fmt_stat(b_stat, b_p, "F")
    ax.annotate(
        "",
        xy=(Y[0] - 0.8, Y[1] + 0.6),
        xytext=(M[0] + 1, M[1] - 0.6),
        arrowprops=dict(arrowstyle="-|>", color=b_col, lw=b_lw),
    )
    ax.text(
        7.0,
        5.9,
        f"Path b\n{b_txt}\np = {b_p:.3f}",
        fontsize=10,
        color=b_col,
        fontweight=b_wt,
        ha="center",
        va="center",
        linespacing=1.3,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8),
    )

    c_txt, c_col, c_wt, c_lw = fmt_stat(c_stat, c_p, "t")
    ax.annotate(
        "",
        xy=(Y[0] - 1.2, Y[1]),
        xytext=(X[0] + 1.2, X[1]),
        arrowprops=dict(arrowstyle="-|>", color=c_col, lw=c_lw, linestyle="--"),
    )
    ax.text(
        5,
        2.2,
        f"Path c'\n{c_txt}\np = {c_p:.3f}",
        fontsize=10,
        color=c_col,
        fontweight=c_wt,
        ha="center",
        va="center",
        linespacing=1.3,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8),
    )


def to_plot_data(coefs: dict) -> dict:
    return {
        "a_coef": coefs["a"]["coef"],
        "a_p": coefs["a"]["pval"] or 1.0,
        "b_coef": coefs["b"]["coef"],
        "b_p": coefs["b"]["pval"] or 1.0,
        "c_coef": coefs["direct"]["coef"],
        "c_p": coefs["direct"]["pval"] or 1.0,
        "indirect": coefs["indirect"]["coef"],
        "indirect_p": coefs["indirect"].get("pval") or 1.0,
    }


def main() -> None:
    df = load_and_prepare_data(
        dosing_path=project_root / "dat" / "Socio_Econ" / "raw" / "current_dosing.csv",
        delivery_path=project_root / "dat" / "Socio_Econ" / "raw" / "delivery_strategy.csv",
    )
    results = run_full_analysis(df, n_boot=5000, seed=42)

    mediation_data = {
        "Delivery": to_plot_data(results["delivery_regression"]["coefficients"]),
        "Dosing": to_plot_data(results["dosing_regression"]["coefficients"]),
    }

    delivery_stats = traditional_mediation_test(df, "HPV_PRIM_DELIV_STRATEGY")
    dosing_stats = traditional_mediation_test(df, "HPV_INT_DOSES")

    fig_t, axes_t = plt.subplots(1, 2, figsize=(14, 5))
    draw_mediation_traditional(axes_t[0], "Delivery Strategy", delivery_stats)
    axes_t[0].set_title("Delivery Strategy as Mediator", fontsize=14, fontweight="bold", pad=45)
    draw_mediation_traditional(axes_t[1], "Dosing Schedule", dosing_stats)
    axes_t[1].set_title("Dosing Schedule as Mediator", fontsize=14, fontweight="bold", pad=45)

    fig_t.text(
        0.5,
        0.01,
        "Note: Green paths indicate statistical significance (p < .05)",
        ha="center",
        fontsize=11,
        color="#333",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out_path_t = Path(__file__).parent / "fig_socio_mediation_traditional.png"
    fig_t.savefig(out_path_t, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig_t)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    draw_mediation_beta(axes[0], "Delivery Strategy", mediation_data["Delivery"])
    axes[0].set_title("Model 1: Delivery Strategy as Mediator", fontsize=14, fontweight="bold", pad=10)
    draw_mediation_beta(axes[1], "Dosing Schedule", mediation_data["Dosing"])
    axes[1].set_title("Model 2: Dosing Schedule as Mediator", fontsize=14, fontweight="bold", pad=10)

    fig.text(
        0.5,
        0.02,
        "Note: Green path indicates statistical significance",
        ha="center",
        fontsize=10,
        color="#333",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = Path(__file__).parent / "fig_socio_mediation_beta.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
