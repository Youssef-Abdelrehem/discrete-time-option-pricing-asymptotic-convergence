# Discrete-Time Option Pricing & Asymptotic Convergence

**Team 4 · Quantitative Financial Modelling 2026 · Project 3**

A from-scratch Python implementation of binomial tree option pricers (CRR and Jarrow-Rudd), with convergence analysis, production benchmarking, and model-risk assessment against the Black-Scholes-Merton closed form.

**Interactive Animation:** [CRR vs JR vs Richardson — Convergence to BSM](https://fil126.github.io/discrete-time-option-pricing-asymptotic-convergence/graphs/convergence_comparison.html)

---

## Repository Structure

```
├── codes/
│   ├── crr_engine.py          # Core pricing engine
│   ├── convergence_plot.py    # Visualization module
│   └── main.py                # Entry point — runs all four parts
├── graphs/                    # Output figures (PNG, 300 DPI) and interactive HTML
│   ├── convergence_comparison.png
│   ├── convergence_comparison.html  # Interactive animation (Plotly)
│   ├── eu_threshold_scan.png
│   └── crr_jr_model_risk_prices.png
└── reference paper/
    └── QFM_Paper.pdf          # Reference paper (D1)
```

---

## Modules

### `crr_engine.py` — Pricing Engine

Contains four functions, all operating on the same parameter signature `(S0, K, T, r, sigma, ...)`:

| Function | Description |
|---|---|
| `bsm_price` | Black-Scholes-Merton closed-form price (European only). Used as benchmark. |
| `crr_price` | Cox-Ross-Rubinstein binomial tree. Supports European and American exercise. O(N²) time, O(N) memory via in-place backward induction. |
| `jr_price` | Jarrow-Rudd binomial tree. Equal risk-neutral probabilities (q = 0.5), drift correction embedded in jump sizes. Same backward induction logic as CRR. |
| `richardson_price` | Richardson extrapolation: `2·CRR(2N) − CRR(N)`. Reduces error from O(1/√N) to O(1/N). Requires even N. |

**CRR parametrization:**
```
u = exp(σ√Δt),  d = 1/u,  q = (exp(rΔt) − d) / (u − d)
```

**JR parametrization:**
```
u = exp((r − σ²/2)Δt + σ√Δt),  d = exp((r − σ²/2)Δt − σ√Δt),  q = 0.5
```

American options use early-exercise enforcement at every node during backward induction:
```python
V = max(continuation_value, intrinsic_value)
```

---

### `convergence_plot.py` — Visualization

| Function | Output | Description |
|---|---|---|
| `convergence_analysis` | dict | Computes prices and absolute/relative errors vs BSM for a list of N values. |
| `plot_convergence_comparison` | `convergence_comparison.png` | Two-panel figure: price trajectory + log-log error decay for CRR, JR, and Richardson on the same axes. |
| `plot_threshold_scan` | `eu_threshold_scan.png` | Semi-log error scan showing the first stable N where all subsequent relative errors stay below 0.05%. |
| `plot_crr_jr_model_risk_prices` | `crr_jr_model_risk_prices.png` | CRR vs JR price convergence with vertical marker at the first stable N where CRR-vs-JR discrepancy drops below 0.02%. |

---

### `main.py` — Entry Point

Runs the full analysis in four parts with reference parameters `S0=100, K=100, T=1, r=0.05, σ=0.20`:

| Part | Description |
|---|---|
| **1A** | Convergence table: CRR, JR, Richardson prices vs BSM for N ∈ {5, 10, 25, 50, 100, 200, 500, 1000}. Includes absolute/relative errors and odd-N Richardson note. |
| **1B** | American vs European comparison for CRR and JR (replicates paper Table 3). Includes N=5000 reference row. |
| **2** | Convergence analysis and log-log OLS slope estimation. Generates comparison plot. |
| **3** | Production validation benchmark: European put first stable N < 0.05% error; American put N=500 vs N=5000 reference. |
| **4** | CRR vs JR model-risk assessment: price discrepancy scan up to N=5000, first stable N where discrepancy < 0.02%. |

---

## Key Results

| Model | Convergence slope | First stable N (< 0.05% vs BSM) |
|---|---|---|
| CRR | −1.000 | N = 725 |
| JR | −0.762 | N = 525 |
| Richardson | −2.007 | — |

- **American put premium** (early exercise): ~0.52 (~9.3% over European) for the reference parameters.
- **CRR vs JR model-risk** stable below 0.02% discrepancy from N = 3000.

---

## Requirements

```
numpy
scipy
matplotlib
plotly
```

Install with:
```bash
pip install -r requirements.txt
```

## Usage

```bash
cd codes
python main.py
```

Generates three PNG figures and one interactive HTML in `graphs/`, and prints all numerical results to stdout.
