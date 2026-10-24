#!/usr/bin/env python3
"""
Generate publication-quality figures for the presentation deck.
Adheres to the "Financial Systems Research Lab" design system:
- Background: #F7F6F2
- Deep Charcoal text: #202722
- Primary Muted Forest Green: #35594D
- Secondary Sage / Neutral: #829688 / #D9D8D0
- Muted Crimson Risk: #B45C52
- Crisp direct labels, no chart clutter.
"""

import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

OUTPUT_DIR = "presentation/assets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Common styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.edgecolor'] = '#D9D8D0'
plt.rcParams['axes.linewidth'] = 0.8

BG_COLOR = '#F7F6F2'
TEXT_COLOR = '#202722'
ACCENT_GREEN = '#35594D'
ACCENT_SAGE = '#829688'
ACCENT_RED = '#B45C52'
CARD_BG = '#FFFFFF'
BORDER_COLOR = '#D9D8D0'


def generate_pr_auc_comparison():
    """Figure 1: Validation PR-AUC across Tabular vs Graph Models."""
    fig, ax = plt.subplots(figsize=(9.2, 5.2), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    models = [
        "Logistic\nRegression",
        "Random\nForest",
        "XGBoost\n(Weighted)",
        "XGBoost + 40\nGraph Feat.",
        "XGBoost\n(Unweighted)",
        "Static\nGraphSAGE",
        "Temporal\nGraphSAGE",
        "Graph\nAttention (GAT)*"
    ]
    scores = [
        0.4126,
        0.4543,
        0.5395,
        0.5440,
        0.5948,
        0.4637,
        0.4927,
        0.3320
    ]
    errors = [
        0.0,
        0.0,
        0.0,
        0.0034, # B subsample SD
        0.0,
        0.0167, # GraphSAGE SD
        0.0035, # Temporal GraphSAGE SD
        0.0052  # GAT SD
    ]
    
    # Colors: Tabular green, Graph sage/red
    colors = [
        '#A4B2A8',
        '#829688',
        '#35594D',
        '#27473C',
        '#1E382F',
        '#829688',
        '#5A786B',
        '#B45C52' # GAT non-converged highlight
    ]

    x = np.arange(len(models))
    bars = ax.bar(x, scores, yerr=errors, capsize=4, width=0.62, color=colors, 
                  edgecolor=TEXT_COLOR, linewidth=0.8, ecolor=TEXT_COLOR)

    # Reference line for baseline XGBoost
    ax.axhline(0.5395, color='#B45C52', linestyle='--', linewidth=1.2, alpha=0.7)
    ax.text(7.4, 0.543, "XGBoost Baseline (0.5395)", color='#B45C52', fontsize=9, 
            fontweight='bold', ha='right', va='bottom')

    # Direct labels above bars
    for i, (bar, score, err) in enumerate(zip(bars, scores, errors)):
        y_pos = score + err + 0.012
        label_text = f"{score:.4f}"
        if i == 7:
            label_text += "\n(Did not converge)"
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos, label_text,
                ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=TEXT_COLOR)

    ax.set_ylim(0.25, 0.66)
    ax.set_ylabel("Validation PR-AUC (Days 122–151)", fontsize=11, fontweight='bold', color=TEXT_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9.5, color=TEXT_COLOR)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(BORDER_COLOR)
    ax.spines['bottom'].set_color(BORDER_COLOR)
    ax.yaxis.grid(True, linestyle=':', alpha=0.6, color=BORDER_COLOR)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "fig1_pr_auc_comparison.png")
    plt.savefig(fig_path, dpi=300, facecolor=BG_COLOR)
    plt.close()
    print(f"Saved: {fig_path}")


def generate_drift_adaptation_chart():
    """Figure 2: 5-Window Adaptive Fine-Tuning vs Frozen vs Replay Control."""
    fig, ax = plt.subplots(figsize=(9.2, 5.2), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    windows = ["Window 1\n(Days 122–127)", "Window 2\n(Days 128–133)", "Window 3\n(Days 134–139)", "Window 4\n(Days 140–145)", "Window 5\n(Days 146–151)"]
    x = np.arange(len(windows))

    # Recorded Phase 8 results (mean of 3 seeds)
    frozen = [0.5868, 0.4003, 0.4210, 0.3782, 0.3350]
    adaptive = [0.5868, 0.4530, 0.4425, 0.4281, 0.3798]
    replay = [0.5868, 0.4303, 0.3918, 0.3300, 0.3504]

    # Plot lines with distinct styling
    ax.plot(x, adaptive, marker='o', markersize=7, linewidth=2.4, color='#35594D', 
            label="Continual Adaptive Fine-Tuning (Mean: 0.4580)")
    ax.plot(x, frozen, marker='s', markersize=7, linewidth=2.0, color='#B45C52', linestyle='--',
            label="Frozen Baseline (Mean: 0.4243)")
    ax.plot(x, replay, marker='^', markersize=7, linewidth=1.8, color='#829688', linestyle=':',
            label="Replay Control (Mean: 0.4179)")

    # Direct labels on final window
    ax.text(4.08, adaptive[-1], f"+0.0448\nGain", color='#35594D', fontsize=9, fontweight='bold', va='center')
    ax.text(4.08, frozen[-1], f"0.3350\n(Severe decay)", color='#B45C52', fontsize=8.5, va='center')

    ax.set_ylim(0.28, 0.64)
    ax.set_ylabel("Window PR-AUC Score", fontsize=11, fontweight='bold', color=TEXT_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(windows, fontsize=9.5, color=TEXT_COLOR)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(BORDER_COLOR)
    ax.spines['bottom'].set_color(BORDER_COLOR)
    ax.yaxis.grid(True, linestyle=':', alpha=0.6, color=BORDER_COLOR)
    ax.set_axisbelow(True)

    ax.legend(frameon=True, facecolor=CARD_BG, edgecolor=BORDER_COLOR, fontsize=9.5, loc='upper right')

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "fig2_drift_adaptation.png")
    plt.savefig(fig_path, dpi=300, facecolor=BG_COLOR)
    plt.close()
    print(f"Saved: {fig_path}")


def generate_class_imbalance_chart():
    """Figure 3: Class Imbalance / Weighting Impact on XGBoost PR-AUC."""
    fig, ax = plt.subplots(figsize=(8.0, 4.8), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    schemes = ["Unweighted\n(scale_pos_weight = 1.0)", "Square-Root\n(scale_pos_weight = 5.23)", "Full Weighting\n(scale_pos_weight = 27.36)"]
    pooled_scores = [0.5948, 0.5792, 0.5395]
    cold_scores = [0.4367, 0.4293, 0.4064]

    x = np.arange(len(schemes))
    width = 0.35

    rects1 = ax.bar(x - width/2, pooled_scores, width, label='Pooled Validation PR-AUC', 
                    color='#35594D', edgecolor=TEXT_COLOR, linewidth=0.8)
    rects2 = ax.bar(x + width/2, cold_scores, width, label='Cold-Start PR-AUC', 
                    color='#829688', edgecolor=TEXT_COLOR, linewidth=0.8)

    # Values on bars
    for rect in rects1:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, h + 0.008, f"{h:.4f}", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#35594D')
    for rect in rects2:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, h + 0.008, f"{h:.4f}", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#5A6E60')

    # Annotate penalty
    ax.annotate("Weighting Penalty:\n-0.0553 PR-AUC",
                xy=(x[2] - width/2, pooled_scores[2]), xytext=(x[1] - 0.1, 0.51),
                arrowprops=dict(arrowstyle="->", color='#B45C52', lw=1.5),
                fontsize=9.5, fontweight='bold', color='#B45C52')

    ax.set_ylim(0.32, 0.65)
    ax.set_ylabel("Validation PR-AUC", fontsize=11, fontweight='bold', color=TEXT_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(schemes, fontsize=9.5, color=TEXT_COLOR)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(BORDER_COLOR)
    ax.spines['bottom'].set_color(BORDER_COLOR)
    ax.yaxis.grid(True, linestyle=':', alpha=0.6, color=BORDER_COLOR)
    ax.set_axisbelow(True)

    ax.legend(frameon=True, facecolor=CARD_BG, edgecolor=BORDER_COLOR, fontsize=9.5, loc='upper right')

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "fig3_class_weighting.png")
    plt.savefig(fig_path, dpi=300, facecolor=BG_COLOR)
    plt.close()
    print(f"Saved: {fig_path}")


def generate_system_architecture_diagram():
    """Diagram: End-to-End System Architecture (4 Stages)."""
    fig, ax = plt.subplots(figsize=(12.2, 5.4), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.axis('off')

    stages = [
        {
            "num": "STAGE 1",
            "title": "Ingestion & Causal Preprocessing",
            "items": [
                "IEEE-CIS Transaction Stream",
                "Chronological Split (Days 1–121 Train)",
                "Train-Only Encoders & Median Scalers",
                "Sealed Held-Out (Days 152–182)"
            ],
            "accent": '#35594D'
        },
        {
            "num": "STAGE 2",
            "title": "Dual Representation Engine",
            "items": [
                "470 Native Tabular Features",
                "40 Backward Graph Features (t' < t)",
                "Heterogeneous Dynamic Graph",
                "5 Node Types & 4 Timed Relations"
            ],
            "accent": '#35594D'
        },
        {
            "num": "STAGE 3",
            "title": "Model Evaluation Suite",
            "items": [
                "Tabular Baselines (LR, RF, XGBoost)",
                "XGBoost + Handcrafted Graph Feat.",
                "Static HeteroSAGE & GAT (Causal)",
                "Temporal GraphSAGE (GRU Memory)"
            ],
            "accent": '#35594D'
        },
        {
            "num": "STAGE 4",
            "title": "Intelligence & Drift Control",
            "items": [
                "Continual Fine-Tuning (Phase 8)",
                "Candidate Communities (Phase 9b)",
                "TreeSHAP & Local Explanations",
                "Standard-Library Local Serving API"
            ],
            "accent": '#B45C52'
        }
    ]

    card_width = 2.45
    card_height = 4.2
    start_x = 0.5
    y_pos = 0.6

    for i, s in enumerate(stages):
        x = start_x + i * 2.85
        
        # Shadow/Card box
        box = patches.FancyBboxPatch((x, y_pos), card_width, card_height,
                                     boxstyle="round,pad=0.1,rounding_size=0.15",
                                     facecolor=CARD_BG, edgecolor=BORDER_COLOR, linewidth=1.2)
        ax.add_patch(box)

        # Stage Header Tag
        tag_box = patches.FancyBboxPatch((x + 0.15, y_pos + card_height - 0.45), 1.0, 0.32,
                                         boxstyle="round,pad=0.05,rounding_size=0.08",
                                         facecolor=s["accent"], edgecolor="none")
        ax.add_patch(tag_box)
        ax.text(x + 0.65, y_pos + card_height - 0.29, s["num"], color='#FFFFFF',
                fontsize=8.5, fontweight='bold', ha='center', va='center')

        # Title
        ax.text(x + 0.15, y_pos + card_height - 0.85, s["title"], color=TEXT_COLOR,
                fontsize=10.5, fontweight='bold', ha='left', va='top', wrap=True)

        # Line separator
        ax.plot([x + 0.15, x + card_width - 0.15], [y_pos + card_height - 1.35, y_pos + card_height - 1.35],
                color=BORDER_COLOR, linewidth=0.8)

        # Bullets
        bullet_y = y_pos + card_height - 1.75
        for item in s["items"]:
            ax.plot(x + 0.22, bullet_y + 0.08, marker='o', markersize=3.5, color=s["accent"])
            ax.text(x + 0.38, bullet_y, item, color='#3C4540', fontsize=8.8, va='bottom', wrap=True)
            bullet_y -= 0.55

        # Connecting Arrow between cards
        if i < len(stages) - 1:
            arrow_x = x + card_width + 0.08
            arrow_y = y_pos + card_height / 2
            ax.annotate("", xy=(arrow_x + 0.24, arrow_y), xytext=(arrow_x, arrow_y),
                        arrowprops=dict(arrowstyle="->", color='#829688', lw=2.0))

    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 5.4)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "diag_system_architecture.png")
    plt.savefig(fig_path, dpi=300, facecolor=BG_COLOR)
    plt.close()
    print(f"Saved: {fig_path}")


def generate_graph_schema_diagram():
    """Diagram: 5 Node Types and 4 Timed Relational Edges."""
    fig, ax = plt.subplots(figsize=(11.0, 5.6), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.axis('off')

    # Central Transaction Node
    center_x, center_y = 5.5, 2.9
    t_box = patches.FancyBboxPatch((center_x - 1.45, center_y - 0.75), 2.9, 1.5,
                                   boxstyle="round,pad=0.1,rounding_size=0.15",
                                   facecolor='#35594D', edgecolor=TEXT_COLOR, linewidth=1.5)
    ax.add_patch(t_box)
    ax.text(center_x, center_y + 0.3, "TRANSACTION NODE", color='#FFFFFF', fontsize=11.5, fontweight='bold', ha='center')
    ax.text(center_x, center_y - 0.15, "590,540 Entities • 470 Features\nTimestamped Event (Seconds Offset)", 
            color='#EFEFEA', fontsize=8.5, ha='center')

    # 4 Entity Nodes around center
    entities = [
        {
            "name": "PSEUDO-ACCOUNT",
            "desc": "199,625 Nodes\ncard1 + addr1 + (day - D1)\nCoverage: 88.9%",
            "rel": "charged_to",
            "pos": (2.0, 4.4),
            "edge_start": (center_x - 1.3, center_y + 0.65),
            "edge_end": (3.1, 4.2),
            "rel_label_pos": (3.8, 4.2)
        },
        {
            "name": "PAYMENT CARD",
            "desc": "13,553 Nodes\ncard1 .. card6 attributes\nCoverage: 100%",
            "rel": "used_card",
            "pos": (9.0, 4.4),
            "edge_start": (center_x + 1.3, center_y + 0.65),
            "edge_end": (7.9, 4.2),
            "rel_label_pos": (7.2, 4.2)
        },
        {
            "name": "NETWORK PROXY",
            "desc": "2,394 Nodes\nP_emaildomain / proxy cluster\nCoverage: 23.6%",
            "rel": "via_network",
            "pos": (2.0, 1.4),
            "edge_start": (center_x - 1.3, center_y - 0.65),
            "edge_end": (3.1, 1.6),
            "rel_label_pos": (3.8, 1.8)
        },
        {
            "name": "DEVICE FINGERPRINT",
            "desc": "4,880 Nodes\nHardware id_30 .. id_33\nCoverage: 12.0%",
            "rel": "from_device",
            "pos": (9.0, 1.4),
            "edge_start": (center_x + 1.3, center_y - 0.65),
            "edge_end": (7.9, 1.6),
            "rel_label_pos": (7.2, 1.8)
        }
    ]

    for e in entities:
        bx, by = e["pos"]
        box = patches.FancyBboxPatch((bx - 1.35, by - 0.65), 2.7, 1.3,
                                     boxstyle="round,pad=0.1,rounding_size=0.12",
                                     facecolor=CARD_BG, edgecolor=BORDER_COLOR, linewidth=1.2)
        ax.add_patch(box)
        ax.text(bx, by + 0.25, e["name"], color='#202722', fontsize=10, fontweight='bold', ha='center')
        ax.text(bx, by - 0.2, e["desc"], color='#606C64', fontsize=8.0, ha='center')

        # Connector arrow
        ax.annotate("", xy=e["edge_end"], xytext=e["edge_start"],
                    arrowprops=dict(arrowstyle="<->", color='#829688', lw=1.8, linestyle='-'))

        # Relation Label Pill
        rx, ry = e["rel_label_pos"]
        ax.text(rx, ry, f"[{e['rel']}]", color='#35594D', fontsize=8.5, fontweight='bold', ha='center',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='#F7F6F2', edgecolor='#D9D8D0', lw=0.6))

    # Footnote with comfortable margin
    ax.text(5.5, 0.45, "Rigorous Exclusion: Billing address, email domain, and raw device strings were measured and excluded\n(produced massive hubs with zero predictive lift above random chance).",
            color='#606C64', fontsize=8.5, ha='center', style='italic')

    ax.set_xlim(0, 11.0)
    ax.set_ylim(0, 5.6)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "diag_graph_schema.png")
    plt.savefig(fig_path, dpi=300, facecolor=BG_COLOR)
    plt.close()
    print(f"Saved: {fig_path}")


def generate_temporal_memory_diagram():
    """Diagram: Streaming Day-by-Day GRU Entity Memory Evolution."""
    fig, ax = plt.subplots(figsize=(10.5, 3.8), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.axis('off')

    days = ["Day t - 1", "Day t", "Day t + 1"]
    start_x = 1.2
    y_pos = 1.6

    for i, d in enumerate(days):
        x = start_x + i * 3.2

        # Daily Snapshot Container
        box = patches.FancyBboxPatch((x - 1.1, y_pos - 0.8), 2.2, 2.3,
                                     boxstyle="round,pad=0.1,rounding_size=0.12",
                                     facecolor=CARD_BG, edgecolor=BORDER_COLOR, linewidth=1.2)
        ax.add_patch(box)
        ax.text(x, y_pos + 1.25, d, color=TEXT_COLOR, fontsize=10.5, fontweight='bold', ha='center')

        # Entity Memory State (GRU)
        gru_box = patches.FancyBboxPatch((x - 0.85, y_pos + 0.35), 1.7, 0.6,
                                         boxstyle="round,pad=0.05,rounding_size=0.08",
                                         facecolor='#35594D', edgecolor='none')
        ax.add_patch(gru_box)
        ax.text(x, y_pos + 0.65, f"h_{i} (64-dim)", color='#FFFFFF', fontsize=9.2, fontweight='bold', ha='center')
        ax.text(x, y_pos + 0.44, "Recurrent Latent State", color='#EFEFEA', fontsize=7.2, ha='center')

        # Day transactions
        txn_box = patches.FancyBboxPatch((x - 0.85, y_pos - 0.6), 1.7, 0.7,
                                         boxstyle="round,pad=0.05,rounding_size=0.08",
                                         facecolor='#F7F6F2', edgecolor='#D9D8D0')
        ax.add_patch(txn_box)
        ax.text(x, y_pos - 0.32, f"Daily Trans. Mask\nEdges <= {d}", color='#202722', fontsize=7.8, ha='center')

        # Vertical connection within day
        ax.annotate("", xy=(x, y_pos + 0.35), xytext=(x, y_pos + 0.12),
                    arrowprops=dict(arrowstyle="->", color='#35594D', lw=1.5))

        # Horizontal recurrent propagation
        if i < len(days) - 1:
            arrow_x = x + 0.85
            arrow_y = y_pos + 0.65
            ax.annotate("", xy=(arrow_x + 1.45, arrow_y), xytext=(arrow_x, arrow_y),
                        arrowprops=dict(arrowstyle="->", color='#35594D', lw=2.0))
            ax.text(arrow_x + 0.72, arrow_y + 0.20, "GRU Update\n+ Δt Elapsed", color='#35594D',
                    fontsize=7.8, fontweight='bold', ha='center')

    # Explanatory tag
    ax.text(5.25, 0.28, "Strict Invariant: Entity states carry forward continuously without label leakage.\nPredictions on day t use h_{t-1} and same-day structure without accessing future days or labels.",
            color='#606C64', fontsize=8.0, ha='center', style='italic')

    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 3.8)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "diag_temporal_memory.png")
    plt.savefig(fig_path, dpi=300, facecolor=BG_COLOR)
    plt.close()
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    generate_pr_auc_comparison()
    generate_drift_adaptation_chart()
    generate_class_imbalance_chart()
    generate_system_architecture_diagram()
    generate_graph_schema_diagram()
    generate_temporal_memory_diagram()
    print("All presentation figures generated successfully.")
