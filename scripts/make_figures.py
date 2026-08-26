#!/usr/bin/env python3
"""Render the result figures for the medical GateMem sweep.

Reads every ``outputs/*/summary.json`` produced by ``bench/scripts/run_eval.py``
and writes light- and dark-mode PNGs into ``docs/figures/``. The two model
families plotted are Qwen/Qwen3.8-27B and Qwen/Qwen2.5-32B-Instruct; unless a
figure says otherwise the numbers are the GPT-4.1-judged ones, which is the only
judge both families share.

    python scripts/make_figures.py
"""

import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "outputs"
FIG_DIR = REPO / "docs" / "figures"

# ---------------------------------------------------------------- palette ----
# Reference data-viz palette. Only categorical slots 1-3 are used: those are the
# slots documented as clearing the all-pairs CVD and normal-vision floors in
# both modes, so no chart here puts an unvalidated pair on screen.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "page": "#f9f9f7",
        "primary": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "baseline": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
    },
    "dark": {
        "surface": "#1a1a19",
        "page": "#0d0d0d",
        "primary": "#ffffff",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "baseline": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70"],
    },
}

FONT = ["DejaVu Sans", "system-ui", "sans-serif"]

# ------------------------------------------------------------------ data ----
MODELS = {
    "qwen38_27b": "Qwen3.8-27B",
    "qwen32b": "Qwen2.5-32B-Instruct",
}

# Fixed display order, so a reader comparing two panels is comparing rows.
AGENTS = [
    ("long_context", "Long-Context"),
    ("rag_naive", "RAG (naive)"),
    ("rag_policy", "RAG (policy)"),
    ("a_mem", "A-Mem"),
    ("mem0", "Mem0"),
    ("remem_i", "ReMeM (iterative)"),
    ("remem_s", "ReMeM (single)"),
]

RUN_RE = re.compile(
    r"^medical_(?P<model>qwen38_27b|qwen32b)_(?P<agent>.+?)(?:__judge-(?P<judge>\w+))?$"
)


def load_runs():
    """Return {(model, agent, judge): summary dict}. judge is 'gpt41' or 'rule'."""
    runs = {}
    for summary in sorted(OUT_DIR.glob("*/summary.json")):
        m = RUN_RE.match(summary.parent.name)
        if not m:
            continue  # smoke tests, contaminated re-runs, pilots
        agent = m.group("agent")
        if agent not in dict(AGENTS):
            continue
        judge = m.group("judge") or "rule"
        runs[(m.group("model"), agent, judge)] = json.loads(summary.read_text())
    return runs


def pct(summary, key):
    return summary.get(key, 0.0) * 100.0


# ---------------------------------------------------------------- drawing ----
def _radii(ax, px):
    """Convert a pixel radius into (rx, ry) data units for this axes.

    Bars are drawn in data coordinates whose x and y units are unrelated, so a
    single radius would come out as a smear in one direction. Layout is fixed
    (no constrained layout, and bbox_inches="tight" crops rather than scales),
    so the axes' pixel size is already known here.
    """
    fig = ax.figure
    box = ax.get_position()
    w_px = box.width * fig.get_figwidth() * fig.dpi
    h_px = box.height * fig.get_figheight() * fig.dpi
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    return px * abs(x1 - x0) / w_px, px * abs(y1 - y0) / h_px


def rounded_bar(ax, ax_y, length, thickness, color, rx, ry):
    """A horizontal bar squared off at the baseline and rounded at the data end.

    matplotlib's own rounded boxes round all four corners, which detaches the
    bar from its axis; the data end is the only end that should be soft.
    """
    rx = max(0.0, min(rx, abs(length)))
    ry = max(0.0, min(ry, thickness / 2.0))
    y0, y1 = ax_y, ax_y + thickness
    x0, x1 = 0.0, length
    verts = [
        (x0, y0), (x1 - rx, y0),
        (x1, y0), (x1, y0 + ry),
        (x1, y1 - ry),
        (x1, y1), (x1 - rx, y1),
        (x0, y1), (x0, y0),
    ]
    codes = [
        MplPath.MOVETO, MplPath.LINETO,
        MplPath.CURVE3, MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CURVE3, MplPath.CURVE3,
        MplPath.LINETO, MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color,
                           edgecolor="none", linewidth=0, zorder=3))


def style_axes(ax, t):
    ax.set_facecolor(t["surface"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, colors=t["muted"], labelsize=9)


def barh_panel(ax, t, labels, values, color, title, subtitle, fmt="{:.1f}"):
    """One horizontal bar panel: direct value labels, no value axis, no grid."""
    n = len(labels)
    vmax = max(values + [1.0])
    ax.set_xlim(0, vmax * 1.30)
    ax.set_ylim(n - 0.5, -0.5)
    style_axes(ax, t)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9.5, color=t["secondary"])
    ax.set_xticks([])

    rx, ry = _radii(ax, 7)
    thickness = 0.52  # leaves a real gap between neighbouring bars
    for i, v in enumerate(values):
        rounded_bar(ax, i - thickness / 2, v, thickness, color, rx, ry)
        ax.text(v + vmax * 0.025, i, fmt.format(v), va="center", ha="left",
                fontsize=9, color=t["secondary"])

    ax.set_title(title, fontsize=11.5, color=t["primary"], loc="left", pad=28,
                 fontweight="bold")
    ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=8.5,
            color=t["muted"], ha="left", va="bottom")


def grouped_barh_panel(ax, t, labels, series, colors, title, subtitle,
                       fmt="{:.1f}"):
    """Two-series grouped horizontal bars. `series` is [(name, values), ...]."""
    n = len(labels)
    k = len(series)
    vmax = max(max(v) for _, v in series) or 1.0
    ax.set_xlim(0, vmax * 1.34)
    ax.set_ylim(n - 0.5, -0.5)
    style_axes(ax, t)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9.5, color=t["secondary"])
    ax.set_xticks([])

    rx, ry = _radii(ax, 6)
    group = 0.68
    slot = group / k
    thickness = slot * 0.80  # the remainder is the surface gap within a pair
    for j, (_, values) in enumerate(series):
        offset = -group / 2 + j * slot
        for i, v in enumerate(values):
            rounded_bar(ax, i + offset, v, thickness, colors[j], rx, ry)
            ax.text(v + vmax * 0.022, i + offset + thickness / 2, fmt.format(v),
                    va="center", ha="left", fontsize=8, color=t["secondary"])

    ax.set_title(title, fontsize=11.5, color=t["primary"], loc="left", pad=28,
                 fontweight="bold")
    ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=8.5,
            color=t["muted"], ha="left", va="bottom")


def figure(t, figsize):
    fig = plt.figure(figsize=figsize, facecolor=t["page"], dpi=200)
    return fig


def caption(fig, t, text):
    fig.text(0.012, 0.012, text, fontsize=7.8, color=t["muted"],
             ha="left", va="bottom")


def save(fig, name, mode):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}_{mode}.png"
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight",
                pad_inches=0.26)
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPO)}")


SOURCE = "GateMem medical split - 579 checkpoints per run (210 utility, 192 privacy, 177 deletion)."

# ------------------------------------------------------------------ figures --
METRICS = [
    ("utility_accuracy", "Utility accuracy", "higher is better", 0),
    ("compliance_utility_score", "Memory Governance Score (MGS)", "higher is better", 0),
    ("privacy_leakage_rate", "Privacy leakage rate", "lower is better", 1),
    ("deletion_leakage_rate", "Deletion leakage rate", "lower is better", 1),
]


def fig_overview(runs, model, judge, mode):
    t = THEMES[mode]
    summaries = {a: runs[(model, a, judge)] for a, _ in AGENTS}
    # One row order for all four panels, ranked by the headline metric.
    order = sorted(AGENTS, key=lambda kv: -pct(summaries[kv[0]], "compliance_utility_score"))
    labels = [name for _, name in order]

    fig = figure(t, (11.4, 8.2))
    axes = fig.subplots(2, 2)
    fig.subplots_adjust(hspace=0.44, wspace=0.34, top=0.815, bottom=0.06,
                        left=0.145, right=0.975)

    for ax, (key, title, direction, slot) in zip(axes.flat, METRICS):
        values = [pct(summaries[a], key) for a, _ in order]
        barh_panel(ax, t, labels, values, t["series"][slot], title,
                   f"% - {direction}")

    judge_txt = "GPT-4.1 judge" if judge == "gpt41" else "rule-based scorer"
    fig.suptitle(f"{MODELS[model]} - memory baselines on the medical split",
                 fontsize=15.5, color=t["primary"], x=0.012, ha="left", y=0.975,
                 fontweight="bold")
    fig.text(0.012, 0.928,
             f"Seven memory architectures scored by the {judge_txt}. "
             "Rows are ranked by MGS and hold that order across all four panels.",
             fontsize=9.8, color=t["secondary"], ha="left", va="top")
    caption(fig, t, SOURCE)
    save(fig, f"{model}_overview", mode)


def _place_labels(ax, points, texts, t, obstacles_pts):
    """Put each point's label in the first candidate slot that collides with
    nothing already on the plot.

    Seven points in a benchmark scatter cluster hard, and fixed offsets put
    four labels on top of each other. Candidates are tried in reading-preference
    order (above, below, right, left, then diagonals) and scored by overlap area
    so the worst case degrades gracefully instead of stacking.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    placed = []
    for (x, y), r_pts in zip(points, obstacles_pts):
        pad = r_pts + 7
        placed.append(ax.transData.transform((x, y)))

    # Marker discs are obstacles too; approximate each as a square bbox.
    from matplotlib.transforms import Bbox
    blocked = []
    for (px, py), r_pts in zip(placed, obstacles_pts):
        r_px = r_pts * fig.dpi / 72.0
        blocked.append(Bbox.from_extents(px - r_px, py - r_px, px + r_px, py + r_px))

    candidates = [(0, 1), (0, -1), (1, 0), (-1, 0),
                  (1, 1), (-1, 1), (1, -1), (-1, -1)]
    for (x, y), r_pts, label in zip(points, obstacles_pts, texts):
        pad = r_pts + 8
        best, best_cost = None, None
        for dx, dy in candidates:
            ha = {0: "center", 1: "left", -1: "right"}[dx]
            va = {0: "center", 1: "bottom", -1: "top"}[dy]
            ann = ax.annotate(label, (x, y), textcoords="offset points",
                              xytext=(dx * pad, dy * pad), ha=ha, va=va,
                              fontsize=8.8, color=t["secondary"],
                              linespacing=1.35, zorder=6)
            bb = ann.get_window_extent(renderer).expanded(1.06, 1.20)
            cost = 0.0
            for other in blocked:
                ix = max(0, min(bb.x1, other.x1) - max(bb.x0, other.x0))
                iy = max(0, min(bb.y1, other.y1) - max(bb.y0, other.y0))
                cost += ix * iy
            ax_bb = ax.get_window_extent(renderer)
            # Push back hard on anything that escapes the plot area.
            cost += 40 * max(0, ax_bb.x0 - bb.x0) * bb.height
            cost += 40 * max(0, bb.x1 - ax_bb.x1) * bb.height
            cost += 40 * max(0, ax_bb.y0 - bb.y0) * bb.width
            cost += 40 * max(0, bb.y1 - ax_bb.y1) * bb.width
            if best_cost is None or cost < best_cost:
                if best is not None:
                    best.remove()
                best, best_cost = ann, cost
            else:
                ann.remove()
            if best_cost == 0:
                break
        blocked.append(best.get_window_extent(renderer).expanded(1.06, 1.20))


def fig_tradeoff(runs, model, judge, mode):
    t = THEMES[mode]
    summaries = {a: runs[(model, a, judge)] for a, _ in AGENTS}

    fig = figure(t, (8.8, 6.4))
    ax = fig.subplots()
    fig.subplots_adjust(top=0.795, bottom=0.135, left=0.10, right=0.975)
    style_axes(ax, t)
    ax.grid(True, color=t["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    xs = [pct(summaries[a], "privacy_leakage_rate") for a, _ in AGENTS]
    ys = [pct(summaries[a], "utility_accuracy") for a, _ in AGENTS]
    mgs = [pct(summaries[a], "compliance_utility_score") for a, _ in AGENTS]

    # A scatter has no zero-baseline obligation, and framing on the data keeps
    # the seven points from collapsing into one corner.
    xpad = (max(xs) - min(xs)) * 0.22 + 2
    ypad = (max(ys) - min(ys)) * 0.22 + 2
    ax.set_xlim(max(0, min(xs) - xpad), max(xs) + xpad)
    ax.set_ylim(max(0, min(ys) - ypad), min(100, max(ys) + ypad))
    ax.set_xlabel("Privacy leakage rate (%)  -  lower is better", fontsize=10,
                  color=t["secondary"], labelpad=9)
    ax.set_ylabel("Utility accuracy (%)  -  higher is better", fontsize=10,
                  color=t["secondary"], labelpad=9)

    # Marker area encodes MGS; a 2px surface ring keeps overlaps legible.
    areas = [70 + v * 14 for v in mgs]
    ax.scatter(xs, ys, s=areas, c=t["series"][0], edgecolors=t["surface"],
               linewidths=2, zorder=4)

    radii_pts = [(a ** 0.5) / 2 + 2 for a in areas]
    _place_labels(ax, list(zip(xs, ys)),
                  [f"{name}\nMGS {v:.1f}" for (_, name), v in zip(AGENTS, mgs)],
                  t, radii_pts)

    ax.annotate("better", xy=(0.015, 0.985), xytext=(0.105, 0.905),
                xycoords="axes fraction", textcoords="axes fraction",
                fontsize=9, color=t["muted"], va="center",
                arrowprops=dict(arrowstyle="->", color=t["muted"], lw=1.4))

    fig.suptitle(f"{MODELS[model]} - the utility / privacy trade-off",
                 fontsize=15.5, color=t["primary"], x=0.012, ha="left", y=0.975,
                 fontweight="bold")
    fig.text(0.012, 0.923,
             "Up and to the left is the goal: answer the question without "
             "leaking the private record. Marker area is the MGS.",
             fontsize=9.8, color=t["secondary"], ha="left", va="top")
    caption(fig, t, SOURCE)
    save(fig, f"{model}_tradeoff", mode)


def fig_leakage_layers(runs, model, judge, mode):
    """Answer-level leakage vs what the retriever actually put in context."""
    t = THEMES[mode]
    summaries = {a: runs[(model, a, judge)] for a, _ in AGENTS}
    labels = [name for _, name in AGENTS]

    fig = figure(t, (11.4, 6.0))
    axes = fig.subplots(1, 2)
    fig.subplots_adjust(wspace=0.34, top=0.735, bottom=0.09, left=0.135,
                        right=0.975)

    for ax, (ans, ctx, title) in zip(axes, [
        ("privacy_leakage_rate", "privacy_context_leakage_rate", "Privacy"),
        ("deletion_leakage_rate", "deletion_context_leakage_rate", "Deletion / staleness"),
    ]):
        grouped_barh_panel(
            ax, t, labels,
            [("Leaked in the answer", [pct(summaries[a], ans) for a, _ in AGENTS]),
             ("Present in the retrieved context",
              [pct(summaries[a], ctx) for a, _ in AGENTS])],
            [t["series"][0], t["series"][1]],
            f"{title} leakage", "% - lower is better")

    handles = [plt.Line2D([], [], marker="s", linestyle="none", markersize=9,
                          color=t["series"][i]) for i in (0, 1)]
    fig.legend(handles, ["Leaked in the answer", "Present in the retrieved context"],
               loc="upper left", bbox_to_anchor=(0.012, 0.875), frameon=False,
               ncol=2, fontsize=9.5, labelcolor=t["secondary"],
               handletextpad=0.6, columnspacing=2.0)

    fig.suptitle(f"{MODELS[model]} - suppression at the answer hides exposure in the context",
                 fontsize=15.5, color=t["primary"], x=0.012, ha="left", y=0.975,
                 fontweight="bold")
    fig.text(0.012, 0.930,
             "The answer model is told the private or deleted record and declines "
             "to repeat it. The record was still retrieved.",
             fontsize=9.8, color=t["secondary"], ha="left", va="top")
    caption(fig, t, SOURCE)
    save(fig, f"{model}_leakage_layers", mode)


def fig_model_comparison(runs, mode):
    t = THEMES[mode]
    labels = [name for _, name in AGENTS]

    fig = figure(t, (11.4, 8.2))
    axes = fig.subplots(2, 2)
    fig.subplots_adjust(hspace=0.44, wspace=0.34, top=0.815, bottom=0.06,
                        left=0.145, right=0.975)

    for ax, (key, title, direction, _) in zip(axes.flat, METRICS):
        grouped_barh_panel(
            ax, t, labels,
            [(MODELS[m], [pct(runs[(m, a, "gpt41")], key) for a, _ in AGENTS])
             for m in ("qwen38_27b", "qwen32b")],
            [t["series"][0], t["series"][1]],
            title, f"% - {direction}")

    handles = [plt.Line2D([], [], marker="s", linestyle="none", markersize=9,
                          color=t["series"][i]) for i in (0, 1)]
    fig.legend(handles, [MODELS["qwen38_27b"], MODELS["qwen32b"]],
               loc="upper left", bbox_to_anchor=(0.012, 0.912), frameon=False,
               ncol=2, fontsize=9.5, labelcolor=t["secondary"],
               handletextpad=0.6, columnspacing=2.0)

    fig.suptitle("Qwen3.8-27B vs Qwen2.5-32B-Instruct - same benchmark, same judge",
                 fontsize=15.5, color=t["primary"], x=0.012, ha="left", y=0.978,
                 fontweight="bold")
    fig.text(0.012, 0.948,
             "All numbers are GPT-4.1-judged, the one judge both model families "
             "were scored under.",
             fontsize=9.8, color=t["secondary"], ha="left", va="top")
    caption(fig, t, SOURCE)
    save(fig, "model_comparison", mode)


def fig_judge_comparison(runs, mode):
    """Qwen3.8-27B is the only family scored under both scorers."""
    t = THEMES[mode]
    labels = [name for _, name in AGENTS]

    fig = figure(t, (11.4, 8.2))
    axes = fig.subplots(2, 2)
    fig.subplots_adjust(hspace=0.44, wspace=0.34, top=0.815, bottom=0.06,
                        left=0.145, right=0.975)

    for ax, (key, title, direction, _) in zip(axes.flat, METRICS):
        grouped_barh_panel(
            ax, t, labels,
            [("GPT-4.1 judge",
              [pct(runs[("qwen38_27b", a, "gpt41")], key) for a, _ in AGENTS]),
             ("Rule-based scorer",
              [pct(runs[("qwen38_27b", a, "rule")], key) for a, _ in AGENTS])],
            [t["series"][0], t["series"][1]],
            title, f"% - {direction}")

    handles = [plt.Line2D([], [], marker="s", linestyle="none", markersize=9,
                          color=t["series"][i]) for i in (0, 1)]
    fig.legend(handles, ["GPT-4.1 judge", "Rule-based scorer"],
               loc="upper left", bbox_to_anchor=(0.012, 0.912), frameon=False,
               ncol=2, fontsize=9.5, labelcolor=t["secondary"],
               handletextpad=0.6, columnspacing=2.0)

    fig.suptitle("Qwen3.8-27B - how much the scorer changes the answer",
                 fontsize=15.5, color=t["primary"], x=0.012, ha="left", y=0.978,
                 fontweight="bold")
    fig.text(0.012, 0.948,
             "The rule-based scorer only credits a literal string match, so it "
             "reads every paraphrase as a miss - and every hedge as a non-leak.",
             fontsize=9.8, color=t["secondary"], ha="left", va="top")
    caption(fig, t, SOURCE)
    save(fig, "qwen38_27b_judge_comparison", mode)


# -------------------------------------------------------------------- table --
def write_table(runs):
    rows = ["| Model | Memory agent | Judge | Utility % | Privacy leak % | "
            "Deletion leak % | MGS % |",
            "|---|---|---|---:|---:|---:|---:|"]
    for model in ("qwen38_27b", "qwen32b"):
        for judge in ("gpt41", "rule"):
            for a, name in AGENTS:
                s = runs.get((model, a, judge))
                if s is None:
                    continue
                rows.append(
                    f"| {MODELS[model]} | {name} | "
                    f"{'GPT-4.1' if judge == 'gpt41' else 'rule-based'} | "
                    f"{pct(s, 'utility_accuracy'):.2f} | "
                    f"{pct(s, 'privacy_leakage_rate'):.2f} | "
                    f"{pct(s, 'deletion_leakage_rate'):.2f} | "
                    f"{pct(s, 'compliance_utility_score'):.2f} |")
    path = FIG_DIR / "results_table.md"
    path.write_text("# Medical split - full results\n\n" + "\n".join(rows) + "\n")
    print(f"  wrote {path.relative_to(REPO)}")


def main():
    plt.rcParams["font.family"] = FONT
    runs = load_runs()
    missing = [k for m in MODELS for a, _ in AGENTS
               for k in [(m, a, "gpt41")] if k not in runs]
    if missing:
        raise SystemExit(f"missing GPT-4.1-judged runs: {missing}")

    for mode in ("light", "dark"):
        print(f"{mode} mode:")
        for model in MODELS:
            fig_overview(runs, model, "gpt41", mode)
            fig_tradeoff(runs, model, "gpt41", mode)
            fig_leakage_layers(runs, model, "gpt41", mode)
        fig_model_comparison(runs, mode)
        fig_judge_comparison(runs, mode)
    write_table(runs)


if __name__ == "__main__":
    main()
