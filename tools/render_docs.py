from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "assets"

COLORS = {
    "ink": "#17212b",
    "muted": "#52606d",
    "line": "#8a99a8",
    "blue": "#d9eaf7",
    "blue_edge": "#277da1",
    "green": "#deefe4",
    "green_edge": "#3f7d5b",
    "yellow": "#fff1c7",
    "yellow_edge": "#b78103",
    "red": "#f8dddd",
    "red_edge": "#a94b4b",
    "violet": "#eadff3",
    "violet_edge": "#77528b",
    "gray": "#eef1f4",
}


def setup(width: float, height: float):
    fig, ax = plt.subplots(figsize=(width, height), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def fitted_text(
    ax,
    x,
    y,
    value,
    *,
    max_width,
    max_height,
    fontsize,
    min_fontsize,
    **kwargs,
):
    artist = ax.text(x, y, value, fontsize=fontsize, **kwargs)
    renderer = ax.figure.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer)

    while artist.get_fontsize() > min_fontsize:
        bbox = artist.get_window_extent(renderer)
        if bbox.width <= axes_bbox.width * max_width and bbox.height <= axes_bbox.height * max_height:
            break
        artist.set_fontsize(max(min_fontsize, artist.get_fontsize() - 0.25))

    bbox = artist.get_window_extent(renderer)
    if bbox.width > axes_bbox.width * max_width or bbox.height > axes_bbox.height * max_height:
        raise ValueError(f"text does not fit its box: {value!r}")
    return artist


def box(ax, x, y, w, h, title, subtitle="", *, fill, edge):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.006,rounding_size=0.008",
        linewidth=1.5,
        facecolor=fill,
        edgecolor=edge,
    )
    ax.add_patch(patch)
    multiline_title = "\n" in title
    fitted_text(
        ax,
        x + w / 2,
        y + h * (0.65 if subtitle and multiline_title else 0.59 if subtitle else 0.50),
        title,
        max_width=w - 0.015,
        max_height=h * (0.46 if multiline_title else 0.34),
        fontsize=11,
        min_fontsize=7.5,
        ha="center",
        va="center",
        fontweight="semibold",
        color=COLORS["ink"],
        linespacing=1.05,
    )
    if subtitle:
        fitted_text(
            ax,
            x + w / 2,
            y + h * (0.22 if multiline_title else 0.29),
            subtitle,
            max_width=w - 0.015,
            max_height=h * (0.34 if multiline_title else 0.42),
            fontsize=8.2,
            min_fontsize=6.4,
            ha="center",
            va="center",
            color=COLORS["muted"],
            linespacing=1.25,
        )
    return patch


def arrow(ax, start, end, *, label="", color=None, style="-|>", curve=0.0):
    color = color or COLORS["line"]
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=12,
        linewidth=1.5,
        color=color,
        connectionstyle=f"arc3,rad={curve}",
    )
    ax.add_patch(patch)
    if label:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2 + (0.035 if curve >= 0 else -0.035)
        ax.text(mx, my, label, ha="center", va="center", fontsize=8, color=COLORS["muted"])
    return patch


def save(fig, name):
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSET_DIR / name, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)


def overview():
    fig, ax = setup(10.8, 4.0)
    nodes = [
        (0.010, "Latent $z$", "Gaussian reference\nconditioned by $N_{FP}$", "violet", "violet_edge"),
        (0.210, "Flow ODE", "$\\dot x=v_\\theta(x,t)$\nforward or inverse", "blue", "blue_edge"),
        (0.410, "Coil tokens", "$x,y,z$ Fourier\ncoefficients + current", "green", "green_edge"),
        (0.610, "GPU evaluator", "axis $\\to s \\to \\psi \\to \\alpha$\nQS volume + engineering", "yellow", "yellow_edge"),
        (0.810, "Score $S$", "larger is better\nstatus + diagnostics", "red", "red_edge"),
    ]
    node_width = 0.17
    for x, title, subtitle, fill, edge in nodes:
        box(ax, x, 0.50, node_width, 0.28, title, subtitle, fill=COLORS[fill], edge=COLORS[edge])
    for (x0, *_), (x1, *_) in zip(nodes[:-1], nodes[1:]):
        arrow(ax, (x0 + node_width, 0.64), (x1, 0.64))

    box(
        ax,
        0.525,
        0.10,
        0.42,
        0.20,
        "Score components",
        "axis | $\\psi$ | surface | coordinates | volume QS | $\\iota$ | coils",
        fill=COLORS["gray"],
        edge=COLORS["line"],
    )
    arrow(ax, (0.70, 0.50), (0.70, 0.30), style="-|>")
    arrow(
        ax,
        (0.90, 0.78),
        (0.10, 0.78),
        label="finite-difference score ascent updates $z$",
        color=COLORS["red_edge"],
        curve=0.08,
    )
    save(fig, "overview.png")


def evaluator():
    fig, ax = setup(11.0, 5.5)
    y = 0.66
    xs = (0.015, 0.18, 0.345, 0.51, 0.675, 0.84)
    labels = [
        ("Coils", "Biot-Savart field", "green", "green_edge"),
        ("Magnetic axis", "Poincare fixed point\n+ elliptic topology", "blue", "blue_edge"),
        ("Fitted $s$", "$\\mathbf{B}\\cdot\\nabla s\\approx0$\nquadratic gauge fixed", "violet", "violet_edge"),
        ("Calibrated $\\psi$", "toroidal flux / $2\\pi$\nouter level screening", "yellow", "yellow_edge"),
        ("$\\alpha,\\iota$ fit", "$\\nabla\\psi\\times\\nabla\\alpha\\approx\\mathbf{B}$\nvector least squares", "blue", "blue_edge"),
        ("Volume QS", "$f_C$ over the volume\nweighted radial bins", "red", "red_edge"),
    ]
    for x, (title, subtitle, fill, edge) in zip(xs, labels):
        box(ax, x, y, 0.14, 0.23, title, subtitle, fill=COLORS[fill], edge=COLORS[edge])
    for x0, x1 in zip(xs[:-1], xs[1:]):
        arrow(ax, (x0 + 0.14, y + 0.115), (x1, y + 0.115))

    box(
        ax,
        0.745,
        0.36,
        0.235,
        0.17,
        "Screening score",
        "7 normalized components + QH gates\nexplicit status on early failure",
        fill=COLORS["gray"],
        edge=COLORS["line"],
    )
    arrow(ax, (0.91, 0.66), (0.86, 0.53), color=COLORS["red_edge"])

    box(
        ax,
        0.14,
        0.08,
        0.18,
        0.19,
        "$\\alpha+\\nu$ initializer",
        "straight-field coordinate\n+ toroidal correction",
        fill=COLORS["green"],
        edge=COLORS["green_edge"],
    )
    box(
        ax,
        0.40,
        0.08,
        0.18,
        0.19,
        "Standard surface",
        "Simsopt least squares\nthen Newton",
        fill=COLORS["yellow"],
        edge=COLORS["yellow_edge"],
    )
    box(
        ax,
        0.66,
        0.08,
        0.18,
        0.19,
        "Validation",
        "dense residual + Poincare\nregularity + DESC",
        fill=COLORS["red"],
        edge=COLORS["red_edge"],
    )
    arrow(ax, (0.58, 0.66), (0.23, 0.27), label="selected candidate surface", curve=0.10)
    arrow(ax, (0.32, 0.175), (0.40, 0.175))
    arrow(ax, (0.58, 0.175), (0.66, 0.175))
    ax.text(0.02, 0.04, "Fast path: fixed work budgets and linear solves.  Full validation is a separate acceptance path.", fontsize=9, color=COLORS["muted"])
    save(fig, "evaluator-pipeline.png")


def optimization():
    fig, ax = setup(10.8, 4.8)
    box(ax, 0.025, 0.58, 0.19, 0.22, "Current latent $z_t$", "one center state", fill=COLORS["violet"], edge=COLORS["violet_edge"])
    box(ax, 0.30, 0.70, 0.16, 0.18, "$z_t+c u_j$", "positive probes", fill=COLORS["blue"], edge=COLORS["blue_edge"])
    box(ax, 0.30, 0.40, 0.16, 0.18, "$z_t-c u_j$", "negative probes", fill=COLORS["blue"], edge=COLORS["blue_edge"])
    box(ax, 0.56, 0.55, 0.16, 0.24, "Flow decode\n+ GPU score", "$x=F(z)$, then $S(x)$\nparallel endpoint batches", fill=COLORS["yellow"], edge=COLORS["yellow_edge"])
    box(ax, 0.80, 0.55, 0.16, 0.24, "Adam ascent", "$\\hat g$ from centered secants\nvalidate center branch", fill=COLORS["green"], edge=COLORS["green_edge"])
    arrow(ax, (0.215, 0.70), (0.30, 0.78), label="+$c u_j$", curve=-0.08)
    arrow(ax, (0.215, 0.66), (0.30, 0.49), label="-$c u_j$", curve=0.08)
    arrow(ax, (0.46, 0.79), (0.56, 0.70))
    arrow(ax, (0.46, 0.49), (0.56, 0.62))
    arrow(ax, (0.72, 0.67), (0.80, 0.67))
    arrow(ax, (0.88, 0.55), (0.12, 0.58), color=COLORS["green_edge"], curve=-0.30)
    ax.text(0.50, 0.33, "accepted $z_{t+1}$", ha="center", va="center", fontsize=8, color=COLORS["green_edge"])

    box(
        ax,
        0.26,
        0.10,
        0.48,
        0.17,
        "Reversible learned coordinates",
        "forward: $x=F(z)$ | inverse: $z=F^{-1}(x)$\nsame ODE integrated in the opposite time direction",
        fill=COLORS["gray"],
        edge=COLORS["line"],
    )
    save(fig, "flow-optimization.png")


def main():
    overview()
    evaluator()
    optimization()


if __name__ == "__main__":
    main()
