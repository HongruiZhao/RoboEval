"""Bar plots of an eval_molmoact2.py run: one figure per task family, grouped by randomization condition.

Each figure has five panels sharing the condition x-axis (static / position / orientation / combined),
one per metric, each on its own raw y-axis: success rate, slips, collisions (env + self), average
cartesian jerk and cartesian path length (both arms). Values are episode means; each metric's y-range is
the same for every task, so panels are comparable across tasks.

Usage (`roboeval` env):
    python evaluation/plot_eval_results.py --episodes <eval_dir>/episodes.jsonl
Writes <eval_dir>/plots/<Family>.png and <eval_dir>/plots/all_tasks.png.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FAMILIES = ["CubeHandover", "VerticalCubeHandover", "LiftPot", "LiftTray", "PackBox",
            "PickSingleBookFromTable", "RotateValve", "StackSingleBookShelf", "StackTwoBlocks"]
SUFFIX_TO_CONDITION = {"": "static", "Position": "position", "Orientation": "orientation",
                       "PositionAndOrientation": "combined", "Obstacle": "obstacle"}
CONDITIONS = ["static", "position", "orientation", "combined"]
# Single-variant tasks that randomize on reset anyway (see RANDOMIZATION_OVERRIDES in the dataset conversion).
CONDITION_OVERRIDES = {"VerticalCubeHandover": "combined"}

# name -> (per-episode value, panel title, value format); colors: validated categorical slots 1-4, fixed order.
METRICS = {
    "success": (lambda r: float(r["success"]), "Success rate", lambda v: f"{v:.0%}"),
    "slips": (lambda r: r["metrics"]["slip_count"], "Slips / episode", lambda v: f"{v:.1f}"),
    "collisions": (lambda r: r["metrics"]["env_collision_count"] + r["metrics"]["self_collision_count"],
                   "Collisions / episode (env + self)", lambda v: f"{v:.1f}"),
    "jerk": (lambda r: r["metrics"]["overall_avg_cartesian_jerk"], "Avg cartesian jerk (m/s³)", lambda v: f"{v:.1f}"),
    "path": (lambda r: r["metrics"]["total_cartesian_path_length"], "Cartesian path length, both arms (m)",
             lambda v: f"{v:.2f}"),
}
COLORS = {"success": "#2a78d6", "slips": "#eb6834", "collisions": "#1baf7a", "jerk": "#eda100", "path": "#e87ba4"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def split_variant(variant: str) -> tuple[str, str]:
    family = max((f for f in FAMILIES if variant.startswith(f)), key=len)
    return family, CONDITION_OVERRIDES.get(variant, SUFFIX_TO_CONDITION[variant[len(family):]])


def plot_panel(ax, family: str, metric: str, table: dict, ymax: float) -> None:
    _, title, fmt = METRICS[metric]
    conditions = CONDITIONS + sorted({c for (f, c) in table if f == family} - set(CONDITIONS))
    x = np.arange(len(conditions))
    for i, cond in enumerate(conditions):
        if (family, cond) not in table:
            ax.text(x[i], ymax * 0.5, "not\nevaluated", ha="center", va="center", fontsize=7, color=INK_2)
            continue
        value = table[(family, cond)][metric]
        ax.bar(x[i], value, 0.6, color=COLORS[metric], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.text(x[i], value + ymax * 0.02, fmt(value), ha="center", va="bottom", fontsize=8, color=INK_2)
    n = {c: table[(family, c)]["episodes"] for c in conditions if (family, c) in table}
    ax.set_xticks(x, [f"{c}\n(n={n[c]})" if c in n else c for c in conditions], color=INK, fontsize=8)
    ax.set_xlim(-0.5, len(conditions) - 0.5)  # keep empty conditions from collapsing the axis
    ax.set_ylim(0, ymax * 1.15)
    if metric == "success":
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title(title, color=INK, fontsize=9, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_2)
    ax.tick_params(axis="y", colors=INK_2, labelsize=8, length=0)


def plot_family_row(axes, family: str, table: dict, ymax: dict) -> None:
    for ax, metric in zip(axes, METRICS):
        plot_panel(ax, family, metric, table, ymax[metric])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True, help="episodes.jsonl from eval_molmoact2.py")
    episodes_path = parser.parse_args().episodes
    rows = [json.loads(line) for line in episodes_path.read_text().splitlines()]

    grouped = defaultdict(list)
    for row in rows:
        grouped[split_variant(row["variant"])].append(row)
    table = {
        group: {"episodes": len(eps), **{m: float(np.mean([value(r) for r in eps])) for m, (value, _, _) in METRICS.items()}}
        for group, eps in grouped.items()
    }
    # One y-range per metric for every task, so panels compare across tasks.
    ymax = {m: max(max(t[m] for t in table.values()), 1e-9) for m in METRICS}
    ymax["success"] = 1.0

    out_dir = episodes_path.parent / "plots"
    out_dir.mkdir(exist_ok=True)
    families = [f for f in FAMILIES if any(g[0] == f for g in table)]
    for family in families:
        fig, axes = plt.subplots(1, len(METRICS), figsize=(20, 3.6), facecolor=SURFACE)
        plot_family_row(axes, family, table, ymax)
        fig.suptitle(family, x=0.01, ha="left", color=INK, fontsize=12)
        fig.tight_layout()
        fig.savefig(out_dir / f"{family}.png", dpi=150, facecolor=SURFACE)
        plt.close(fig)

    fig, axes = plt.subplots(len(families), len(METRICS), figsize=(22, 2.9 * len(families)), facecolor=SURFACE)
    for row_axes, family in zip(axes, families):
        plot_family_row(row_axes, family, table, ymax)
        row_axes[0].annotate(family, xy=(0, 1.22), xycoords="axes fraction", fontsize=11, color=INK,
                             fontweight="bold", ha="left")
    fig.tight_layout(h_pad=2.5)
    fig.savefig(out_dir / "all_tasks.png", dpi=110, facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {len(families)} task plots + all_tasks.png to {out_dir}")


if __name__ == "__main__":
    main()
