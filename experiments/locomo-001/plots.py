"""Draws the LoCoMo locomo-001 figures from results/*.json.

Usage: uv run --with matplotlib python experiments/locomo-001/plots.py
Writes figures/accuracy_by_category.png and figures/cost_latency.png next to this file.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
OURS = "#2a78d6"  # categorical slot 1
ORIGINAL = "#eb6834"  # categorical slot 2
SYSTEMS = (("original", "Original codelab", ORIGINAL), ("pr13", "Tiered memory (PR13)", OURS))
# The original's ledger covers ingestion and answers together; the README splits it 3.39 / 5.96.
ORIGINAL_ANSWER_USD = 5.96
CATEGORIES = (("multi-hop", 1), ("temporal", 2), ("open-domain", 3), ("single-hop", 4))


def load(name):
    return json.loads((RESULTS / name).read_text())


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_2)
    ax.tick_params(colors=INK_2, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def accuracy_by_category(scores):
    rows = scores["per_question"]
    groups = [("All", rows)] + [
        (name, [r for r in rows if r["category"] == cat]) for name, cat in CATEGORIES
    ]
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    style(ax)
    width = 0.36
    for i, (key, label, color) in enumerate(SYSTEMS):
        for g, (_, qs) in enumerate(groups):
            correct = sum(r[key] for r in qs)
            pct = 100 * correct / len(qs)
            x = g + (i - 0.5) * (width + 0.02)
            ax.bar(x, pct, width, color=color, label=label if g == 0 else None)
            ax.text(x, pct + 1.5, f"{pct:.0f}%", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{name}\n(n={len(qs)})" for name, qs in groups], color=INK)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Questions answered correctly (%)", color=INK_2)
    ax.set_title(
        "LoCoMo conv-26, categories 1-4: correct answers by category",
        color=INK,
        loc="left",
        fontsize=11,
    )
    ax.legend(frameon=False, loc="upper right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(FIGURES / "accuracy_by_category.png", dpi=150, facecolor=SURFACE)


def cost_latency(accounting, latency, n_questions):
    # PR13's answers are the run that started from the cached memory; its other row is ingestion.
    pr13 = next(r for r in accounting["rows"] if r["system"] == "pr13" and r["prepared_cache_hit"])
    answer_cost = {
        "pr13": pr13["generation_usd"] / n_questions,
        "original": ORIGINAL_ANSWER_USD / n_questions,
    }
    panels = (
        ("Answer cost per question (USD)", answer_cost, "${:.4f}"),
        (
            "Median answer time per question (s)",
            {k: latency[f"{k}_answer_seconds"]["median"] for k, _, _ in SYSTEMS},
            "{:.1f} s",
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), facecolor=SURFACE)
    for ax, (title, values, fmt) in zip(axes, panels):
        style(ax)
        top = max(values.values())
        for i, (key, label, color) in enumerate(SYSTEMS):
            ax.bar(i, values[key], 0.55, color=color)
            ax.text(
                i,
                values[key] + top * 0.03,
                fmt.format(values[key]),
                ha="center",
                va="bottom",
                fontsize=9,
                color=INK,
            )
        ax.set_xticks(range(len(SYSTEMS)))
        ax.set_xticklabels([label for _, label, _ in SYSTEMS], color=INK, fontsize=9)
        ax.set_ylim(0, top * 1.2)
        ax.set_title(title, color=INK, loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "cost_latency.png", dpi=150, facecolor=SURFACE)


def main():
    FIGURES.mkdir(exist_ok=True)
    scores = load("scores.json")
    accuracy_by_category(scores)
    cost_latency(load("accounting.json"), load("latency.json"), scores["overall"]["n"])


if __name__ == "__main__":
    main()
