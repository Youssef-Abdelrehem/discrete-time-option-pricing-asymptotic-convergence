import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from crr_engine import crr_price, jr_price, bsm_price


def convergence_analysis(S0, K, T, r, sigma, option_type="put", N_values=None, pricer=None):
    """
    Compute prices and errors for each N in N_values using the given pricer.

    pricer: callable with signature f(S0, K, T, r, sigma, N, option_type, exercise).
            Defaults to crr_price.

    Returns a dict with keys: N_values, prices, bsm_ref, errors_abs, errors_rel.
    """
    if N_values is None:
        N_values = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000]
    if pricer is None:
        pricer = crr_price

    bsm_ref = bsm_price(S0, K, T, r, sigma, option_type)

    prices = np.array([
        pricer(S0, K, T, r, sigma, N, option_type, "european")
        for N in N_values
    ])

    errors_abs = np.abs(prices - bsm_ref)
    errors_rel = errors_abs / bsm_ref * 100

    return {
        "N_values":   np.array(N_values),
        "prices":     prices,
        "bsm_ref":    bsm_ref,
        "errors_abs": errors_abs,
        "errors_rel": errors_rel,
    }


def plot_convergence_comparison(results_dict, save_path="convergence_comparison.png"):
    """
    Two-panel comparison figure for multiple models on the same axes.

    results_dict: dict of {label: results} where each results comes from
                  convergence_analysis(). All must share the same N_values and bsm_ref.

    Example:
        plot_convergence_comparison({
            "CRR": crr_results,
            "JR":  jr_results,
        })
    """
    # Colour and marker scheme — one entry per model, extensible
    STYLES = [
        {"color": "#1f4e79", "marker": "o", "linestyle": "-"},   # CRR — dark blue
        {"color": "#c00000", "marker": "s", "linestyle": "-"},   # JR  — dark red
        {"color": "#7030a0", "marker": "^", "linestyle": "-"},   # Richardson — purple
    ]
    RED  = "#c00000"
    GREY = "#7f7f7f"

    # BSM reference is the same for all models
    bsm = next(iter(results_dict.values()))["bsm_ref"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "European Put — CRR vs Jarrow-Rudd vs Richardson: Convergence to Black-Scholes-Merton",
        fontsize=13, fontweight="bold", y=1.01
    )

    ax1, ax2 = axes

    for idx, (label, res) in enumerate(results_dict.items()):
        style  = STYLES[idx % len(STYLES)]
        N      = res["N_values"]
        prices = res["prices"]
        errors = res["errors_abs"]

        # Panel 1 — price trajectory
        ax1.semilogx(N, prices,
                     color=style["color"], linewidth=1.6,
                     marker=style["marker"], markersize=4,
                     linestyle=style["linestyle"],
                     label=label, zorder=3 - idx)

        # Panel 2 — error decay
        ax2.loglog(N, errors,
                   color=style["color"], linewidth=1.6,
                   marker=style["marker"], markersize=4,
                   linestyle=style["linestyle"],
                   label=label, zorder=3 - idx)

    # BSM reference line on panel 1
    ax1.axhline(bsm, color=GREY, linewidth=1.4, linestyle="--",
                label=f"BSM  ${bsm:.4f}$", zorder=1)

    # O(1/sqrt(N)) guide on panel 2, calibrated on the first model
    first_errors = next(iter(results_dict.values()))["errors_abs"]
    first_N      = next(iter(results_dict.values()))["N_values"]
    N_guide      = np.array([first_N.min(), first_N.max()], dtype=float)
    scale        = first_errors[0] * np.sqrt(first_N[0])
    ax2.loglog(N_guide, scale / np.sqrt(N_guide),
               color=GREY, linewidth=1.2, linestyle=":",
               label=r"$\mathcal{O}(N^{-1/2})$ guide", zorder=1)

    # Formatting — panel 1
    ax1.set_xlabel("Number of Time Steps $N$", fontsize=11)
    ax1.set_ylabel("Option Price", fontsize=11)
    ax1.set_title("Convergence Trajectory", fontsize=11)
    ax1.legend(fontsize=10, framealpha=0.9)
    ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax1.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
    ax1.tick_params(labelsize=10)

    # Formatting — panel 2
    ax2.set_xlabel("Number of Time Steps $N$", fontsize=11)
    ax2.set_ylabel("Absolute Pricing Error", fontsize=11)
    ax2.set_title(r"Error Decay  ($\log$-$\log$ scale)", fontsize=11)
    ax2.legend(fontsize=10, framealpha=0.9)
    ax2.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax2.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax2.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
    ax2.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Comparison plot saved: {save_path}")
    return fig


def plot_crr_jr_model_risk_prices(N_values, crr_prices, jr_prices,
                                  threshold_n=None, threshold=0.02,
                                  save_path="crr_jr_model_risk_prices.png"):
    """
    Plot CRR and JR European prices against N, with a vertical line marking
    the first stable N where the CRR-vs-JR discrepancy is below threshold.
    """
    N = np.array(N_values)
    crr_prices = np.array(crr_prices)
    jr_prices = np.array(jr_prices)

    BLUE = "#1f4e79"
    RED  = "#c00000"
    GREY = "#7f7f7f"

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(
        "European Put — CRR vs Jarrow-Rudd Model-Risk Check",
        fontsize=13, fontweight="bold", y=1.01
    )

    ax.semilogx(N, crr_prices, color=BLUE, linewidth=1.7,
                marker="o", markersize=4, label="CRR price")
    ax.semilogx(N, jr_prices, color=RED, linewidth=1.7,
                marker="s", markersize=4, label="JR price")

    if threshold_n is not None:
        ax.axvline(threshold_n, color="#404040", linewidth=1.3,
                   linestyle="--", label=f"Stable discrepancy < {threshold}%")
        ax.annotate(
            f"N={threshold_n}",
            xy=(threshold_n, np.interp(threshold_n, N, crr_prices)),
            xytext=(threshold_n * 1.08, max(crr_prices.max(), jr_prices.max()) - 0.012),
            fontsize=9, color="#404040",
            arrowprops=dict(arrowstyle="->", color="#404040", lw=0.8)
        )

    ax.set_xlabel("Number of Time Steps $N$", fontsize=11)
    ax.set_ylabel("European Put Price", fontsize=11)
    ax.set_title("Price convergence across binomial parametrizations", fontsize=11)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Model-risk price plot saved: {save_path}")
    return fig


def plot_threshold_scan(N_values, errors_pct_dict, threshold=0.05,
                        save_path="eu_threshold_scan.png"):
    """
    Relative-error scan plot for multiple models vs a horizontal threshold line.

    errors_pct_dict: {"CRR": array_of_pct_errors, "JR": array_of_pct_errors, ...}
    threshold: % value drawn as a dashed reference line (default 0.05)
    """
    STYLES = [
        {"color": "#1f4e79", "marker": "o"},
        {"color": "#c00000", "marker": "s"},
        {"color": "#375623", "marker": "^"},
    ]
    GREY = "#7f7f7f"
    N = np.array(N_values)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(
        "European Put — Relative Pricing Error vs BSM Closed-Form",
        fontsize=13, fontweight="bold", y=1.01
    )

    for idx, (label, errs) in enumerate(errors_pct_dict.items()):
        style = STYLES[idx % len(STYLES)]
        errs  = np.array(errs)
        ax.semilogy(N, errs, color=style["color"], linewidth=1.6,
                    marker=style["marker"], markersize=5, label=label, zorder=3 - idx)

        # Mark first STABLE crossing: all subsequent errors also below threshold
        stable_idx = next(
            (i for i in range(len(errs)) if np.all(errs[i:] < threshold)),
            None
        )
        if stable_idx is not None:
            ci = stable_idx
            ax.axvline(N[ci], color=style["color"], linewidth=1.8,
                       linestyle="--", alpha=0.9)
            ax.annotate(
                f"N={N[ci]}\n(stable)",
                xy=(N[ci], errs[ci]),
                xytext=(N[ci] * 1.1, errs[ci] * 5.0),
                fontsize=8, color=style["color"],
                arrowprops=dict(arrowstyle="->", color=style["color"], lw=0.7)
            )

    # Threshold reference line
    ax.axhline(threshold, color=GREY, linewidth=1.3, linestyle="--",
               label=f"Threshold  {threshold}%", zorder=1)
    ax.fill_between(N, 0, threshold, alpha=0.05, color="green",
                    label="Target zone")

    ax.set_xlabel("Number of Time Steps $N$", fontsize=11)
    ax.set_ylabel("Relative Error vs BSM  (%)", fontsize=11)
    ax.set_title("First N achieving < 0.05% relative error", fontsize=11)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f%%"))
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Threshold scan plot saved: {save_path}")
    return fig
