"""
Deliverable 2 - Computational Pricing Prototype
================================================
Team 4 - QFM2026 - Project 3
"""

import os
import numpy as np
from crr_engine import crr_price, jr_price, bsm_price, richardson_price
from convergence_plot import (convergence_analysis,
                              plot_convergence_comparison, plot_threshold_scan,
                              plot_crr_jr_model_risk_prices)
from convergence_animated import plot_convergence_html

# Reference parameters (memo Section 2.2)
S0    = 100.0
K     = 100.0
T     = 1.0
r     = 0.05
sigma = 0.20

N_TABLE = [5, 10, 25, 50, 100, 200, 500, 1000]
N_DENSE           = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000]
N_DENSE_EVEN      = [n for n in N_DENSE if n % 2 == 0]  # Richardson requires even N
N_MODEL_RISK_SCAN = [50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000]

DIR       = os.path.dirname(os.path.abspath(__file__))
GRAPH_DIR = os.path.join(DIR, "..", "graphs")
SEP = "=" * 65


def print_section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")


def convergence_slope(results, min_N=100):
    """OLS slope on log-log space for N >= min_N."""
    mask  = results["N_values"] >= min_N
    ln_N  = np.log(results["N_values"][mask].astype(float))
    ln_E  = np.log(results["errors_abs"][mask])
    return float(np.polyfit(ln_N, ln_E, 1)[0])


def main():
    bsm_eu = bsm_price(S0, K, T, r, sigma, "put")

    # ── Part 1A: European put — CRR vs JR convergence table ──────────────
    print_section("PART 1A - European Put: CRR vs JR Convergence Table")
    print(f"  Parameters: S0={S0}, K={K}, T={T}, r={r}, sigma={sigma}")
    print(f"  BSM reference price: {bsm_eu:.6f}\n")

    hdr = f"  {'N':>6}  {'CRR':>10}  {'JR':>10}  {'Richardson':>10}  {'Err CRR':>9}  {'Err JR':>9}  {'Err Ric':>9}  {'Rel CRR%':>9}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for N in N_TABLE:
        crr = crr_price(      S0, K, T, r, sigma, N, "put", "european")
        jr  = jr_price(       S0, K, T, r, sigma, N, "put", "european")
        ric = richardson_price(S0, K, T, r, sigma, N, "put", "european")
        ec  = abs(crr - bsm_eu)
        ej  = abs(jr  - bsm_eu)
        er  = abs(ric - bsm_eu)
        note = " (*)" if N % 2 != 0 else ""
        print(f"  {N:>6}  {crr:>10.6f}  {jr:>10.6f}  {ric:>10.6f}"
              f"  {ec:>9.6f}  {ej:>9.6f}  {er:>9.6f}  {ec/bsm_eu*100:>8.4f}%{note}")

    print("  (*) Odd N: CRR(N) and CRR(2N) on opposite sides of BSM — Richardson error amplified.")

    # ── Part 1B: American put — CRR vs JR  (replicates paper Table 3) ───
    print_section("PART 1B - American Put: CRR vs JR (Table 3)")
    print(f"  {'N':>6}  {'CRR EU':>10}  {'CRR AM':>10}  {'JR EU':>10}  {'JR AM':>10}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    for N in N_TABLE:
        crr_eu = crr_price(S0, K, T, r, sigma, N, "put", "european")
        crr_am = crr_price(S0, K, T, r, sigma, N, "put", "american")
        jr_eu  = jr_price( S0, K, T, r, sigma, N, "put", "european")
        jr_am  = jr_price( S0, K, T, r, sigma, N, "put", "american")
        print(f"  {N:>6}  {crr_eu:>10.6f}  {crr_am:>10.6f}  {jr_eu:>10.6f}  {jr_am:>10.6f}")

    # N=5000 reference row (paper Table 3, last row — CRR AM was the only value given)
    print(f"  {'------':>6}  {'----------':>10}  {'----------':>10}  {'----------':>10}  {'----------':>10}")
    print("  Computing N=5000 reference row ...")
    ref_crr_eu = crr_price(S0, K, T, r, sigma, 5000, "put", "european")
    ref_crr_am = crr_price(S0, K, T, r, sigma, 5000, "put", "american")
    ref_jr_eu  = jr_price( S0, K, T, r, sigma, 5000, "put", "european")
    ref_jr_am  = jr_price( S0, K, T, r, sigma, 5000, "put", "american")
    print(f"  {'5000':>6}  {ref_crr_eu:>10.6f}  {ref_crr_am:>10.6f}  {ref_jr_eu:>10.6f}  {ref_jr_am:>10.6f}  <- Ref.")

    # ── Part 2: Convergence analysis ─────────────────────────────────────
    print_section("PART 2 - Convergence Analysis")
    print("  Computing convergence for CRR and JR ...")

    crr_results = convergence_analysis(S0, K, T, r, sigma, "put", N_DENSE, pricer=crr_price)
    jr_results  = convergence_analysis(S0, K, T, r, sigma, "put", N_DENSE, pricer=jr_price)
    ric_results = convergence_analysis(S0, K, T, r, sigma, "put", N_DENSE_EVEN, pricer=richardson_price)

    # Comparison plot CRR vs JR vs Richardson
    plot_convergence_comparison(
        {"CRR": crr_results, "JR": jr_results, "Richardson": ric_results},
        save_path=os.path.join(GRAPH_DIR, "convergence_comparison.png")
    )

    # Interactive HTML animation
    plot_convergence_html(
        {"CRR": crr_results, "JR": jr_results, "Richardson": ric_results},
        save_path=os.path.join(GRAPH_DIR, "convergence_comparison.html")
    )

    # Convergence slopes
    crr_slope = convergence_slope(crr_results)
    jr_slope  = convergence_slope(jr_results)
    ric_slope = convergence_slope(ric_results)
    print(f"\n  Convergence slope (log-log OLS, N>=100):")
    print(f"    CRR: {crr_slope:.3f}  |  JR: {jr_slope:.3f}  |  Richardson: {ric_slope:.3f}")
    print(f"    Required <= -0.45  ->  CRR: {'PASS' if crr_slope <= -0.45 else 'CHECK'}"
          f"  |  JR: {'PASS' if jr_slope <= -0.45 else 'CHECK'}"
          f"  |  Richardson: {'PASS' if ric_slope <= -0.45 else 'CHECK'}")

    # ── Part 3: Production Validation Benchmark ──────────────────────────
    print_section("PART 3 - Production Validation Benchmark")

    # 3a. European: scan N values to find when rel error first drops below 0.05%
    print("  3a. European put  — first N achieving < 0.05% rel error vs BSM")
    N_EU_SCAN = [5, 10, 15, 20, 25, 30, 40, 50] + list(range(75, 1025, 25))
    crr_errs_pct, jr_errs_pct = [], []
    print(f"    {'N':>6}  {'CRR err%':>10}  {'':6}  {'JR err%':>10}  {'':6}")
    print(f"    {'-'*6}  {'-'*10}  {'-'*6}  {'-'*10}  {'-'*6}")
    for N in N_EU_SCAN:
        crr_n       = crr_price(S0, K, T, r, sigma, N, "put", "european")
        jr_n        = jr_price( S0, K, T, r, sigma, N, "put", "european")
        err_crr_pct = abs(crr_n - bsm_eu) / bsm_eu * 100
        err_jr_pct  = abs(jr_n  - bsm_eu) / bsm_eu * 100
        crr_errs_pct.append(err_crr_pct)
        jr_errs_pct.append(err_jr_pct)
        tag_crr = "PASS" if err_crr_pct < 0.05 else "    "
        tag_jr  = "PASS" if err_jr_pct  < 0.05 else "    "
        print(f"    {N:>6}  {err_crr_pct:>10.4f}%  {tag_crr}  {err_jr_pct:>10.4f}%  {tag_jr}")

    # First stable N: all subsequent errors in the scan remain below threshold
    def first_stable(n_list, errs, thr=0.05):
        for i in range(len(n_list)):
            if all(e < thr for e in errs[i:]):
                return n_list[i]
        return None

    crr_stable = first_stable(N_EU_SCAN, crr_errs_pct)
    jr_stable  = first_stable(N_EU_SCAN, jr_errs_pct)
    print(f"\n    First stable N (all subsequent < 0.05%):")
    print(f"      CRR -> N={crr_stable if crr_stable else '>1000'}"
          f"  |  JR -> N={jr_stable if jr_stable else '>1000'}")

    plot_threshold_scan(
        N_EU_SCAN,
        {"CRR": crr_errs_pct, "JR": jr_errs_pct},
        threshold=0.05,
        save_path=os.path.join(GRAPH_DIR, "eu_threshold_scan.png")
    )

    # 3b. American: CRR at N=500 vs high-N reference (N=5000, threshold 0.05%)
    print("\n  3b. American put  (N=500 vs N=5000 reference, threshold 0.05%)")
    print("      Computing N=5000 reference  ...")
    ref_am = crr_price(S0, K, T, r, sigma, 5000, "put", "american")
    print(f"      Reference (N=5000): {ref_am:.6f}")
    crr_500_am = crr_price(S0, K, T, r, sigma, 500, "put", "american")
    err_am = abs(crr_500_am - ref_am) / ref_am * 100
    print(f"    CRR N=500: {crr_500_am:.6f}  rel err {err_am:.4f}%  "
          f"{'PASS' if err_am < 0.05 else 'CHECK'}")

    # ── Part 4: JR as Secondary Model / Model-Risk Assessment ────────────
    print_section("PART 4 - CRR vs JR Model-Risk Assessment")
    discrepancy_threshold = 0.02
    crr_mr_prices, jr_mr_prices, discrepancies_pct = [], [], []

    print(f"  European put — CRR-vs-JR discrepancy threshold: {discrepancy_threshold:.2f}%")
    print(f"  {'N':>6}  {'CRR':>10}  {'JR':>10}  {'Discrepancy %':>15}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*15}")

    for N in N_MODEL_RISK_SCAN:
        crr_n = crr_price(S0, K, T, r, sigma, N, "put", "european")
        jr_n  = jr_price( S0, K, T, r, sigma, N, "put", "european")
        disc  = abs(crr_n - jr_n) / crr_n * 100
        crr_mr_prices.append(crr_n)
        jr_mr_prices.append(jr_n)
        discrepancies_pct.append(disc)
        print(f"  {N:>6}  {crr_n:>10.6f}  {jr_n:>10.6f}  {disc:>14.4f}%")

    stable_idx = next(
        (i for i in range(len(discrepancies_pct))
         if all(d < discrepancy_threshold for d in discrepancies_pct[i:])),
        None
    )
    stable_n = N_MODEL_RISK_SCAN[stable_idx] if stable_idx is not None else None

    if stable_n is None:
        print(f"\n  First stable N: not reached in scan up to N={N_MODEL_RISK_SCAN[-1]}")
    else:
        print(f"\n  First stable N: {stable_n}"
              f"  (all subsequent discrepancies < {discrepancy_threshold:.2f}%)")

    plot_crr_jr_model_risk_prices(
        N_MODEL_RISK_SCAN, crr_mr_prices, jr_mr_prices,
        threshold_n=stable_n, threshold=discrepancy_threshold,
        save_path=os.path.join(GRAPH_DIR, "crr_jr_model_risk_prices.png")
    )

    print(f"\n{SEP}\n  Done.\n{SEP}\n")


if __name__ == "__main__":
    main()
