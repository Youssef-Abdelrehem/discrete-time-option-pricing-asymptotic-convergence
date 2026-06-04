import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_convergence_html(results_dict, save_path="convergence_comparison.html"):
    """
    Two-phase interactive animation (Plotly HTML):
      On load   : only BSM reference and O(N^-1/2) guide visible; no curves drawn.
      Phase 1   : "▶ CRR & JR"      — CRR and JR build up progressively.
      Phase 2   : "▶ Add Richardson" — Richardson appears while CRR/JR stay frozen.
      Pause     : stops whichever phase is running.
    Smooth cubic-in-out easing between frames.
    """
    COLORS  = {"CRR": "#1f4e79", "JR": "#c00000", "Richardson": "#7030a0"}
    SYMBOLS = {"CRR": "circle",  "JR": "square",  "Richardson": "triangle-up"}
    GREY = "#7f7f7f"

    crr_res = results_dict["CRR"]
    jr_res  = results_dict["JR"]
    ric_res = results_dict["Richardson"]
    bsm     = float(crr_res["bsm_ref"])

    all_N_vals = sorted(set(
        int(n) for res in results_dict.values() for n in res["N_values"]
    ))
    x_min, x_max = float(all_N_vals[0]), float(all_N_vals[-1])

    guide_x = np.logspace(np.log10(x_min), np.log10(x_max), 150)
    scale   = float(crr_res["errors_abs"][0]) * np.sqrt(float(crr_res["N_values"][0]))
    guide_y = scale / np.sqrt(guide_x)

    # ── Axis ranges (computed early — needed in frame construction) ───────────
    all_prices  = np.concatenate([crr_res["prices"], jr_res["prices"], ric_res["prices"]])
    p_margin    = (all_prices.max() - all_prices.min()) * 0.15
    price_range = [all_prices.min() - p_margin, all_prices.max() + p_margin]

    log_x_min = np.floor(np.log10(x_min) - 0.1)
    log_x_max = np.ceil(np.log10(x_max)  + 0.1)

    # Phase 1 error range: CRR + JR + guide
    p1_errors      = np.concatenate([crr_res["errors_abs"], jr_res["errors_abs"], guide_y])
    p1_errors      = p1_errors[p1_errors > 0]
    log_err_min_p1 = np.floor(np.log10(p1_errors.min()) - 0.3)
    log_err_max_p1 = np.ceil(np.log10(p1_errors.max())  + 0.3)

    # Phase 2 error range: expand downward to show Richardson's much smaller errors
    ric_errors_nz  = ric_res["errors_abs"][ric_res["errors_abs"] > 1e-12]
    log_err_min_p2 = np.floor(np.log10(ric_errors_nz.min()) - 0.3) if len(ric_errors_nz) else log_err_min_p1 - 4
    log_err_max_p2 = log_err_max_p1

    # ── Figure ───────────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Convergence Trajectory", "Error Decay (log–log)"),
        horizontal_spacing=0.11,
    )

    # ── Static traces (indices 0, 1 — always visible) ────────────────────────
    fig.add_trace(go.Scatter(
        x=[x_min, x_max], y=[bsm, bsm],
        mode="lines",
        line=dict(color="#404040", width=1.5, dash="dash"),
        name=f"BSM  {bsm:.4f}",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=guide_x.tolist(), y=guide_y.tolist(),
        mode="lines",
        line=dict(color=GREY, width=1.2, dash="dot"),
        name="O(N⁻¹/²) guide",
    ), row=1, col=2)

    # ── Animated traces (indices 2–7) — all start empty ─────────────────────
    # Richardson also starts with showlegend=False so it is absent from the
    # legend until the first p2 frame sets showlegend=True.
    for label, is_richardson in [("CRR", False), ("JR", False), ("Richardson", True)]:
        color  = COLORS[label]
        symbol = SYMBOLS[label]

        fig.add_trace(go.Scatter(           # prices — col 1
            x=[], y=[],
            mode="lines+markers",
            line=dict(color=color, width=1.8),
            marker=dict(symbol=symbol, size=6, color=color),
            name=label,
            legendgroup=label,
            showlegend=not is_richardson,   # Richardson hidden until p2 starts
        ), row=1, col=1)

        fig.add_trace(go.Scatter(           # errors — col 2
            x=[], y=[],
            mode="lines+markers",
            line=dict(color=color, width=1.8),
            marker=dict(symbol=symbol, size=6, color=color),
            name=label,
            legendgroup=label,
            showlegend=False,
        ), row=1, col=2)

    # Trace index map:
    #   0 BSM line   1 guide
    #   2 CRR prices 3 CRR errors
    #   4 JR prices  5 JR errors
    #   6 Ric prices 7 Ric errors

    # ── Phase 1 frames: CRR & JR build up ────────────────────────────────────
    p1_N = sorted(set(
        int(n) for res in [crr_res, jr_res] for n in res["N_values"]
    ))
    p1_frames = []
    for n_max in p1_N:
        mc = crr_res["N_values"] <= n_max
        mj = jr_res["N_values"]  <= n_max
        p1_frames.append(go.Frame(
            name=f"p1_{n_max}",
            traces=[2, 3, 4, 5],
            data=[
                go.Scatter(x=crr_res["N_values"][mc].tolist(), y=crr_res["prices"][mc].tolist()),
                go.Scatter(x=crr_res["N_values"][mc].tolist(), y=crr_res["errors_abs"][mc].tolist()),
                go.Scatter(x=jr_res["N_values"][mj].tolist(),  y=jr_res["prices"][mj].tolist()),
                go.Scatter(x=jr_res["N_values"][mj].tolist(),  y=jr_res["errors_abs"][mj].tolist()),
            ],
        ))

    # ── Phase 2 frames: Richardson builds up ─────────────────────────────────
    p2_N = sorted(int(n) for n in ric_res["N_values"])
    p2_frames = []
    for i, n_max in enumerate(p2_N):
        mr = ric_res["N_values"] <= n_max
        # First frame: expand error axis downward + reveal Richardson in legend
        first = (i == 0)
        frame_layout = go.Layout(yaxis2=dict(range=[log_err_min_p2, log_err_max_p2])) if first else None
        p2_frames.append(go.Frame(
            name=f"p2_{n_max}",
            traces=[6, 7],
            data=[
                go.Scatter(
                    x=ric_res["N_values"][mr].tolist(),
                    y=ric_res["prices"][mr].tolist(),
                    showlegend=first,
                ),
                go.Scatter(
                    x=ric_res["N_values"][mr].tolist(),
                    y=ric_res["errors_abs"][mr].tolist(),
                ),
            ],
            layout=frame_layout,
        ))

    fig.frames = p1_frames + p2_frames

    # ── Animation timing ─────────────────────────────────────────────────────
    smooth = dict(
        frame=dict(duration=630, redraw=True),
        transition=dict(duration=470, easing="cubic-in-out"),
        fromcurrent=False,
        mode="immediate",
    )
    pause_args = [[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]

    # ── Buttons ──────────────────────────────────────────────────────────────
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            direction="left",
            x=0.5, xanchor="center",
            y=-0.13,
            showactive=True,
            buttons=[
                dict(label="▶  CRR & JR",
                     method="animate",
                     args=[[f.name for f in p1_frames], smooth]),
                dict(label="▶  Add Richardson",
                     method="animate",
                     args=[[f.name for f in p2_frames], smooth]),
                dict(label="⏸  Pause",
                     method="animate",
                     args=pause_args),
            ],
        )],
    )

    # ── Axes and layout ──────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text="European Put — CRR vs Jarrow-Rudd vs Richardson: Convergence to BSM",
            font=dict(size=14),
            x=0.5,
        ),
        legend=dict(x=1.02, y=1, borderwidth=1),
        height=530,
        margin=dict(l=65, r=160, t=75, b=110),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=11),
    )

    fig.update_xaxes(type="log", showgrid=True, gridcolor="#e8e8e8",
                     title_text="Number of Time Steps N", minor_showgrid=True,
                     range=[log_x_min, log_x_max])
    fig.update_yaxes(showgrid=True, gridcolor="#e8e8e8",
                     title_text="Option Price", row=1, col=1,
                     range=price_range)
    # Error panel starts with Phase 1 range
    fig.update_yaxes(type="log", showgrid=True, gridcolor="#e8e8e8",
                     title_text="Absolute Pricing Error", row=1, col=2,
                     minor_showgrid=True,
                     range=[log_err_min_p1, log_err_max_p1])

    fig.write_html(save_path, include_plotlyjs="cdn", auto_play=False)
    print(f"Animation saved: {save_path}")
    return fig
