#!/usr/bin/env python3
"""Original plots using figures4papers style conventions; see figures/README.md."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from verify_research_summary import load_effects, load_path_effects, verify

OUT = Path(__file__).resolve().parents[1] / "figures"
BLUE, TEAL, RED = "#0F4D92", "#42949E", "#B64342"


def apply_publication_style():
    plt.rcParams.update({
        "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 15, "axes.linewidth": 2,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "svg.hashsalt": "cyclediffusion-research",
    })


def finalize_figure(fig, name):
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        metadata = {"CreationDate": None} if ext == "pdf" else ({"Date": None} if ext == "svg" else {})
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight",
                    pad_inches=0.16, metadata=metadata)
        if ext == "svg":
            path = OUT / f"{name}.{ext}"
            path.write_text("\n".join(line.rstrip() for line in
                            path.read_text().splitlines()) + "\n")
    plt.close(fig)


def reliability():
    data = load_effects()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    colors, markers = [BLUE, TEAL, RED], ["o", "s", "D"]
    for ax, metric, title, direction in zip(
        axes, ("identity_delta", "robust_content_error"),
        ("A  Speaker identity", "B  Robust content error"),
        ("Higher favors candidate", "Lower favors candidate"),
    ):
        ax.axvline(0, color="#767676", lw=1.4, ls="--", zorder=0)
        for i, (_, metrics) in enumerate(data):
            mean, low, high = metrics[metric]
            y = 2 - i
            ax.errorbar(mean, y, xerr=[[mean-low], [high-mean]], fmt=markers[i],
                        color=colors[i], capsize=5, ms=9, lw=2.3)
            ax.text(mean, y + .18, f"{mean:+.5f} [{low:+.5f}, {high:+.5f}]",
                    ha="center", va="bottom", fontsize=11)
        ax.set_yticks([2, 1, 0], ["Discovery\njoint", "Discovery\ncontent-only",
                                "Confirmation\ncontent-only"])
        ax.set_ylim(-.5, 2.7)
        ax.set_title(title, loc="left", weight="bold", pad=16)
        ax.set_xlabel("Candidate minus uniform\n" + direction)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim((-.027, .065) if metric == "identity_delta" else (-.115, .055))
    fig.tight_layout(pad=2.0, w_pad=3.0)
    fig.text(.5, -.02, "Stored 95% intervals: discovery, source-cluster bootstrap; confirmation, seed-then-source-unit bootstrap.",
             ha="center", fontsize=11)
    finalize_figure(fig, "reliability_effects")


def path_length():
    data = load_path_effects()
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    for i, (_, (mean, low, high)) in enumerate(data):
        y = 1-i
        ax.errorbar(mean, y, xerr=[[mean-low], [high-mean]], fmt=["o", "s"][i],
                    color=[BLUE, RED][i], ms=9, lw=2.3, capsize=5)
        ax.text(mean, y+.18, f"{mean:+.5f} [{low:+.5f}, {high:+.5f}]",
                ha="center", fontsize=12)
    ax.axvline(0, color="#767676", lw=1.4, ls="--")
    ax.set_yticks([1, 0], ["Word edit distance\nfrom direct anchor",
                         "Signed robust content\nerror change"])
    ax.set_xlim(-.018, .175)
    ax.set_ylim(-.5, 1.6)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Longer paths increase content divergence", loc="left", weight="bold", pad=18)
    ax.set_xlabel("Three-hop minus two-hop effect (higher is worse)")
    fig.tight_layout(pad=2)
    fig.text(.5, -.02, "420 outputs; 60 paired units; 20 source clusters sharing four speakers. Stored 95% cluster-bootstrap intervals.",
             ha="center", fontsize=10)
    finalize_figure(fig, "path_length_effects")


def evidence_chain():
    fig, ax = plt.subplots(figsize=(16, 4.6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    stages = [
        ("1  Reproduction", "Reference sensitivity\nIdentity-content trade-off", BLUE),
        ("2  Gradient path", "Historical snapshot check\nDifferentiable second leg", BLUE),
        ("3  Controlled ablation", "Components + cycle strength\nFreeze content-only, lambda 1", TEAL),
        ("4  Confirmation", "New cohort; three seeds\nMixed identity replication", RED),
        ("5  Path diagnosis", "Two-hop vs. three-hop\nDiscovery cohort revisited", BLUE),
    ]
    for i, (title, body, color) in enumerate(stages):
        x = .12 + i*3.2
        ax.add_patch(FancyBboxPatch((x, 1.45), 2.95, 1.75,
                     boxstyle="round,pad=0.06,rounding_size=0.1",
                     facecolor="#F7F8FA", edgecolor=color, linewidth=2))
        ax.text(x+1.475, 2.76, title, ha="center", va="center", color=color,
                weight="bold", fontsize=14)
        ax.text(x+1.475, 2.08, body, ha="center", va="center", fontsize=11.5,
                linespacing=1.65)
        if i < 4:
            ax.annotate("", xy=(x+3.17, 2.3), xytext=(x+3.02, 2.3),
                        arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#767676"})
    ax.text(.06, 3.82, "CycleDiffusion: from reproduction to bounded research evidence",
            weight="bold", fontsize=20)
    ax.text(.06, .78, "Research sequence; arrows do not denote a shared dataset or a single evolving model.", fontsize=12)
    ax.text(.06, .28, "Discovery: historical checkpoint exposure possible.  Confirmation: fixed evaluation IDs excluded from new base and continuation training.",
            fontsize=11, color="#4D4D4D")
    finalize_figure(fig, "research_evidence_chain")


if __name__ == "__main__":
    verify()
    apply_publication_style()
    evidence_chain()
    reliability()
    path_length()
