"""
GP-Kalman Filter — interactive step-through viewer
====================================================
Provides the ``StepViewer`` class: a Matplotlib figure that lets the user
step through one time-step at a time using the ← / → arrow keys or the
Prev / Next buttons.

Usage
-----
    from viewer import StepViewer
    import matplotlib.pyplot as plt

    viewer = StepViewer(
        frame_ids, meas, scores, gt_pos, gt_vis, records,
        window_size=..., n_forecast=...,
        joint_idx=..., joint_name=..., coord=..., sequence=...,
    )
    plt.show()
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

_VIEW_HISTORY = 45   # frames shown to the left of the current step

_COLORS = {
    "gt":       "#888888",
    "raw_meas": "tomato",
    "posterior": "steelblue",
    "gp_mean":  "darkorange",
    "gp_band":  "orange",
    "window":   "steelblue",
    "cur_post": "steelblue",
    "cur_meas": "red",
}


class StepViewer:
    """
    Matplotlib-based step-through viewer for GP-Kalman filter records.

    Parameters
    ----------
    frame_ids  : sorted list of frame identifiers (ints)
    meas       : raw measurements per frame (float or None)
    scores     : keypoint confidence scores per frame
    gt_pos     : ground-truth coordinate per frame (float or None)
    gt_vis     : ground-truth visibility flag per frame (0 or 1)
    records    : list of per-step dicts produced by ``run_filter``
    window_size : sliding-window length N  (used for start index + label)
    n_forecast  : number of forecast frames shown to the right
    joint_idx   : COCO joint index (for axis label)
    joint_name  : human-readable joint name (for title / label)
    coord       : ``"x"`` or ``"y"``  (for axis label)
    sequence    : sequence name shown in the figure title
    """

    def __init__(
        self,
        frame_ids: list,
        meas: list,
        scores: list,
        gt_pos: list,
        gt_vis: list,
        records: list,
        *,
        window_size: int,
        n_forecast: int,
        joint_idx: int,
        joint_name: str,
        coord: str,
        sequence: str,
    ) -> None:
        self.frame_ids  = frame_ids
        self.meas       = meas
        self.scores     = scores
        self.gt_pos     = gt_pos
        self.gt_vis     = gt_vis
        self.records    = records
        self.n          = len(records)
        self.window_size = window_size
        self.n_forecast  = n_forecast
        self.joint_idx   = joint_idx
        self.joint_name  = joint_name
        self.coord       = coord.upper()
        self.sequence    = sequence

        # Start at the first fully-populated window
        self.idx = min(window_size, self.n - 1)

        self._build_figure()
        self._draw()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_figure(self) -> None:
        self.fig = plt.figure(figsize=(17, 8))
        self.fig.suptitle(
            f"GP-Kalman Filter  ·  {self.sequence}  ·  "
            f"joint {self.joint_idx}: {self.joint_name}  ·  "
            f"coord {self.coord}  ·  window N={self.window_size}",
            fontsize=11, fontweight="bold",
        )

        self.ax      = self.fig.add_axes([0.05, 0.20, 0.65, 0.72])
        self.ax_info = self.fig.add_axes([0.72, 0.20, 0.26, 0.72])
        self.ax_info.axis("off")

        ax_prev = self.fig.add_axes([0.28, 0.04, 0.13, 0.09])
        ax_next = self.fig.add_axes([0.44, 0.04, 0.13, 0.09])
        self.btn_prev = Button(ax_prev, "◀  Prev", color="#e8e8e8", hovercolor="#d0d0d0")
        self.btn_next = Button(ax_next, "Next  ▶", color="#d0e8ff", hovercolor="#b0d0f0")
        self.btn_prev.on_clicked(lambda _: self._move(-1))
        self.btn_next.on_clicked(lambda _: self._move(+1))

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    # ── Navigation ────────────────────────────────────────────────────────────

    def _on_key(self, event) -> None:
        if event.key in ("right", "l"):
            self._move(+1)
        elif event.key in ("left", "h"):
            self._move(-1)

    def _move(self, delta: int) -> None:
        new = max(0, min(self.n - 1, self.idx + delta))
        if new != self.idx:
            self.idx = new
            self._draw()

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw(self) -> None:
        ax  = self.ax
        rec = self.records[self.idx]
        t   = rec["t"]
        ax.cla()

        t_lo = max(self.frame_ids[0],  t - _VIEW_HISTORY)
        t_hi = min(self.frame_ids[-1], t + self.n_forecast + 5)
        fids = self.frame_ids

        # Ground truth
        gt_t = [fids[k] for k in range(len(fids))
                if t_lo <= fids[k] <= t_hi and self.gt_vis[k] > 0]
        gt_y = [self.gt_pos[k] for k in range(len(fids))
                if t_lo <= fids[k] <= t_hi and self.gt_vis[k] > 0]
        if gt_t:
            ax.plot(gt_t, gt_y, lw=1.2, color=_COLORS["gt"], alpha=0.7,
                    zorder=1, label="GT")
            ax.scatter(gt_t, gt_y, s=14, c=_COLORS["gt"], alpha=0.5, zorder=2)

        # Raw measurements
        m_t = [fids[k] for k in range(len(fids))
               if t_lo <= fids[k] <= t_hi and self.meas[k] is not None]
        m_y = [self.meas[k] for k in range(len(fids))
               if t_lo <= fids[k] <= t_hi and self.meas[k] is not None]
        if m_t:
            ax.scatter(m_t, m_y, marker="x", s=28, c=_COLORS["raw_meas"],
                       alpha=0.45, zorder=3, label="Raw measurement")

        # Filter posterior history
        hist_t = [self.records[k]["t"] for k in range(self.idx + 1)
                  if t_lo <= self.records[k]["t"] <= t]
        hist_y = [self.records[k]["mu_post"] for k in range(self.idx + 1)
                  if t_lo <= self.records[k]["t"] <= t]
        if hist_t:
            ax.plot(hist_t, hist_y, lw=2.0, color=_COLORS["posterior"],
                    zorder=5, label="Filter posterior")

        # GP forward projection (pre-computed)
        if rec["fwd_t"] and rec["fwd_mus"] is not None:
            fwd_t    = rec["fwd_t"]
            fwd_mus  = rec["fwd_mus"]
            fwd_stds = rec["fwd_stds"]
            ax.plot(fwd_t, fwd_mus, "--", lw=1.6, color=_COLORS["gp_mean"],
                    zorder=4, label="GP mean (forecast)")
            ax.fill_between(
                fwd_t,
                fwd_mus - 2 * fwd_stds,
                fwd_mus + 2 * fwd_stds,
                color=_COLORS["gp_band"], alpha=0.18, zorder=0,
                label="GP ±2σ (forecast)",
            )

        # Window highlight
        T_post = rec["T_post"]
        if T_post:
            ax.axvspan(min(T_post), max(T_post), alpha=0.08,
                       color=_COLORS["window"],
                       label=f"Window (N={self.window_size})")

        # Current-step markers
        ax.scatter([t], [rec["mu_post"]], s=110, c=_COLORS["cur_post"],
                   zorder=7, label="Posterior (now)",
                   edgecolors="white", linewidths=0.8)
        if rec["z"] is not None:
            ax.scatter([t], [rec["z"]], s=130, c=_COLORS["cur_meas"],
                       marker="*", zorder=8, label="Measurement (now)")
        ax.axvline(t, lw=0.9, ls=":", color="k", alpha=0.55)

        ax.set_xlim(t_lo - 1, t_hi + 1)
        ax.set_xlabel("Frame index", fontsize=9)
        ax.set_ylabel(f"{self.joint_name}  {self.coord}-coordinate  [px]", fontsize=9)
        ax.set_title(f"Step {self.idx} / {self.n - 1}   (frame t = {t})", fontsize=10)
        ax.legend(fontsize=7, loc="upper left", ncol=2, framealpha=0.85)
        ax.grid(True, alpha=0.22)

        self._draw_info(rec)
        # draw() + flush_events() prevents the macOS freeze caused by draw_idle()
        # deferring the render past the end of the button/key callback.
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _draw_info(self, rec: dict) -> None:
        ax = self.ax_info
        ax.cla()
        ax.axis("off")

        sp = rec["sigma_pred"]
        st = rec["sigma_post"]
        inn = rec["innovation"]
        mp = rec["mu_pred"]
        mt = rec["mu_post"]

        lines = [
            f" t = {rec['t']}   (step {self.idx})",
            "",
            f" Joint  : {self.joint_name}",
            f" Coord  : {self.coord}",
            f" Window : N = {self.window_size}",
            f" Buffer : {len(rec['T_post'])} / {self.window_size} pts",
            "",
            "──── Prediction ─────────────────",
            f"  μ_pred  = {mp:>9.3f}  px",
            f"  σ²_pred = {sp:>9.4f}  px²",
            f"  σ_pred  = {sp**0.5:>9.3f}  px",
            "",
            "──── Measurement ────────────────",
        ]

        if rec["z"] is not None:
            R = rec["R"]
            lines += [
                f"  z       = {rec['z']:>9.3f}  px",
                f"  score   = {rec['score']:>9.4f}",
                f"  R (σ²)  = {R:>9.4f}  px²",
                f"  R (σ)   = {R**0.5:>9.3f}  px",
                f"  innovation = {inn:>9.3f}  px",
            ]
        else:
            lines.append("  (no measurement)")

        lines += [
            "",
            "──── Posterior ──────────────────",
            f"  μ_post  = {mt:>9.3f}  px",
            f"  σ²_post = {st:>9.4f}  px²",
            f"  σ_post  = {st**0.5:>9.3f}  px",
            "",
            "──── Kalman gain ────────────────",
        ]
        if rec["z"] is not None and rec["R"] is not None:
            K_gain = sp / (sp + rec["R"])
            lines.append(f"  K       = {K_gain:>9.4f}")
        else:
            lines.append("  K       =       —")

        lines += ["", "  ← / → to step through frames"]

        ax.text(
            0.03, 0.98, "\n".join(lines),
            transform=ax.transAxes, va="top", ha="left",
            fontfamily="monospace", fontsize=8.5,
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#fffff0",
                edgecolor="#cccc88",
                alpha=0.92,
            ),
        )
