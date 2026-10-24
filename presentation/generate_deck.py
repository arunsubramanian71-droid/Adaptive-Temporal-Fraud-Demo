#!/usr/bin/env python3
"""
Generate the Master PowerPoint Presentation:
"Adaptive Temporal Graph Intelligence for Fraud Detection"
Style: "Financial Systems Research Lab" (Restrained, Editorial, Academic)
Target: 16:9 Widescreen (.pptx)
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

OUTPUT_DIR = "presentation"
ASSETS_DIR = "presentation/assets"
PPTX_PATH = os.path.join(OUTPUT_DIR, "Adaptive_Temporal_Fraud_Detection.pptx")

# Palette constants
C_BG = RGBColor(247, 246, 242)        # #F7F6F2 Warm Ivory
C_CARD_BG = RGBColor(255, 255, 255)   # #FFFFFF Crisp White
C_TEXT_MAIN = RGBColor(32, 39, 34)    # #202722 Deep Charcoal
C_TEXT_MUTED = RGBColor(96, 108, 100) # #606C64 Muted Slate
C_ACCENT_GREEN = RGBColor(53, 89, 77) # #35594D Muted Forest Green
C_ACCENT_SAGE = RGBColor(130, 150, 136)# #829688 Restrained Sage
C_ACCENT_RED = RGBColor(180, 92, 82)  # #B45C52 Muted Crimson Risk
C_BORDER = RGBColor(217, 216, 208)    # #D9D8D0 Neutral Border
C_DARK_BG = RGBColor(32, 39, 34)      # #202722 Dark Alternative

FONT_PRIMARY = "Arial"


def add_slide_background(slide, prs=None, dark=False):
    """Adds a full-bleed background shape."""
    w = prs.slide_width if prs is not None else Inches(13.333333)
    h = prs.slide_height if prs is not None else Inches(7.5)
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, w, h)
    bg.fill.solid()
    bg.fill.fore_color.rgb = C_DARK_BG if dark else C_BG
    bg.line.fill.background()
    return bg



def add_header(slide, kicker_text, title_text, subtitle_text, dark=False):
    """Standardized academic research header."""
    tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.48), Inches(11.733), Inches(1.35))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    # Kicker / Section Category
    p_kicker = tf.paragraphs[0]
    p_kicker.text = kicker_text.upper()
    p_kicker.font.name = FONT_PRIMARY
    p_kicker.font.size = Pt(10)
    p_kicker.font.bold = True
    p_kicker.font.color.rgb = C_ACCENT_SAGE if dark else C_ACCENT_GREEN
    p_kicker.space_after = Pt(3)

    # Title
    p_title = tf.add_paragraph()
    p_title.text = title_text
    p_title.font.name = FONT_PRIMARY
    p_title.font.size = Pt(23)
    p_title.font.bold = True
    p_title.font.color.rgb = C_BG if dark else C_TEXT_MAIN
    p_title.space_after = Pt(4)

    # Subtitle / Thesis
    p_sub = tf.add_paragraph()
    p_sub.text = subtitle_text
    p_sub.font.name = FONT_PRIMARY
    p_sub.font.size = Pt(12.5)
    p_sub.font.color.rgb = C_BORDER if dark else C_TEXT_MUTED


def add_card(slide, left, top, width, height, bg_color=C_CARD_BG, border_color=C_BORDER):
    """Creates a card container."""
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = bg_color
    card.line.color.rgb = border_color
    card.line.width = Pt(1)
    return card


def set_speaker_notes(slide, notes_text):
    """Sets detailed speaker notes for delivery and defense."""
    notes_slide = slide.notes_slide
    tf = notes_slide.notes_text_frame
    tf.text = notes_text


def build_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1: TITLE SLIDE (Cover)
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    add_slide_background(s1, prs, dark=True)

    # Decorative thin accent line
    line = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.1), Inches(1.5), Inches(0.04))
    line.fill.solid()
    line.fill.fore_color.rgb = C_ACCENT_SAGE
    line.line.fill.background()

    tb = s1.shapes.add_textbox(Inches(0.8), Inches(1.35), Inches(11.733), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p_k = tf.paragraphs[0]
    p_k.text = "FINANCIAL SYSTEMS RESEARCH LAB • ACADEMIC EVALUATION STUDY"
    p_k.font.name = FONT_PRIMARY
    p_k.font.size = Pt(11)
    p_k.font.bold = True
    p_k.font.color.rgb = C_ACCENT_SAGE
    p_k.space_after = Pt(14)

    p_t = tf.add_paragraph()
    p_t.text = "Adaptive Temporal Graph Intelligence\nfor Fraud Detection"
    p_t.font.name = FONT_PRIMARY
    p_t.font.size = Pt(38)
    p_t.font.bold = True
    p_t.font.color.rgb = C_BG
    p_t.space_after = Pt(14)

    p_sub = tf.add_paragraph()
    p_sub.text = "Evaluating Relational Structure, Temporal Entity Memory, and Continual Adaptation on Streaming Transactions"
    p_sub.font.name = FONT_PRIMARY
    p_sub.font.size = Pt(16)
    p_sub.font.color.rgb = RGBColor(190, 195, 190)
    p_sub.space_after = Pt(36)

    # Presenters Card Box on Title
    card1 = add_card(s1, Inches(0.8), Inches(4.8), Inches(5.6), Inches(1.8), bg_color=RGBColor(42, 51, 45), border_color=C_ACCENT_GREEN)
    tb_c1 = s1.shapes.add_textbox(Inches(1.05), Inches(4.95), Inches(5.1), Inches(1.5))
    tf_c1 = tb_c1.text_frame
    tf_c1.word_wrap = True
    p1 = tf_c1.paragraphs[0]
    p1.text = "PRESENTER A  [Foundations & Design]"
    p1.font.bold = True
    p1.font.size = Pt(12)
    p1.font.color.rgb = C_ACCENT_SAGE
    p1_sub = tf_c1.add_paragraph()
    p1_sub.text = "Problem Formulation • Causal Safeguards • Architecture\nHeterogeneous Graph Construction • Temporal Memory Model"
    p1_sub.font.size = Pt(10.5)
    p1_sub.font.color.rgb = C_BG

    card2 = add_card(s1, Inches(6.8), Inches(4.8), Inches(5.6), Inches(1.8), bg_color=RGBColor(42, 51, 45), border_color=C_ACCENT_GREEN)
    tb_c2 = s1.shapes.add_textbox(Inches(7.05), Inches(4.95), Inches(5.1), Inches(1.5))
    tf_c2 = tb_c2.text_frame
    tf_c2.word_wrap = True
    p2 = tf_c2.paragraphs[0]
    p2.text = "PRESENTER B  [Implementation & Evidence]"
    p2.font.bold = True
    p2.font.size = Pt(12)
    p2.font.color.rgb = C_ACCENT_SAGE
    p2_sub = tf_c2.add_paragraph()
    p2_sub.text = "Evaluation Protocol • Tabular vs Graph Findings • Ablations\nConcept Drift Adaptation • Candidate Communities • Integrity"
    p2_sub.font.size = Pt(10.5)
    p2_sub.font.color.rgb = C_BG

    # Bottom notice
    tb_note = s1.shapes.add_textbox(Inches(0.8), Inches(6.8), Inches(11.733), Inches(0.4))
    tf_note = tb_note.text_frame
    pn = tf_note.paragraphs[0]
    pn.text = "Rigorous validation-period empirical study on the IEEE-CIS Fraud Detection benchmark (590,540 transactions, 182-day span)."
    pn.font.size = Pt(9.5)
    pn.font.color.rgb = C_ACCENT_SAGE

    set_speaker_notes(s1, """[PRESENTER A - OPENING]
Welcome everyone. Today my co-presenter and I are presenting our research project: 'Adaptive Temporal Graph Intelligence for Fraud Detection.'

Our investigation asks a fundamental question in financial machine learning: Does modeling complex relational graphs and temporal behavioral memory actually improve fraud detection over highly tuned tabular baselines like XGBoost, especially when tested under strict real-world temporal causality and concept drift?

To deliver a thorough technical presentation, we have divided the talk into two natural, balanced sections:
- I will deliver Part I, covering the problem formulation, our causal leakage safeguards, the system architecture, our graph schema, and temporal memory modeling.
- My co-presenter will deliver Part II, covering our experimental evaluation protocol, empirical benchmark results, drift adaptation findings, explainability modules, and scientific conclusions.

Let us begin with the core challenge of financial transaction fraud.""")

    # =========================================================================
    # SLIDE 2: THE CORE CHALLENGE (Presenter A)
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_slide_background(s2, prs)
    add_header(s2, "PART I — FOUNDATIONS & SYSTEM DESIGN  •  PRESENTER A",
               "Transaction Fraud: Camouflage, Non-Stationarity, and Coordination",
               "Why isolated transaction classification fails against sophisticated financial adversaries")

    # Left Container: Three Fundamental Dilemmas
    add_card(s2, Inches(0.8), Inches(2.0), Inches(5.7), Inches(4.9))
    tb_l = s2.shapes.add_textbox(Inches(1.05), Inches(2.2), Inches(5.2), Inches(4.5))
    tf_l = tb_l.text_frame
    tf_l.word_wrap = True

    p = tf_l.paragraphs[0]
    p.text = "Three Core Adversarial Dilemmas"
    p.font.bold = True
    p.font.size = Pt(14)
    p.font.color.rgb = C_TEXT_MAIN
    p.space_after = Pt(14)

    items_l = [
        ("1. Deceptive Camouflage", 
         "Individual fraudulent transactions deliberately mimic legitimate user spending in amounts, merchant categories, and purchase timing to bypass static rules."),
        ("2. Temporal Non-Stationarity & Drift", 
         "Fraud patterns evolve rapidly. Adversaries alter strategies across weeks, causing static detection models to suffer substantial decay in production."),
        ("3. Distributed Coordination", 
         "Organized rings distribute attacks across multiple cards, devices, and network proxies to keep velocity beneath account-level alert thresholds.")
    ]
    for title, desc in items_l:
        pt = tf_l.add_paragraph()
        pt.text = title
        pt.font.bold = True
        pt.font.size = Pt(12)
        pt.font.color.rgb = C_ACCENT_GREEN
        pd = tf_l.add_paragraph()
        pd.text = desc
        pd.font.size = Pt(10.5)
        pd.font.color.rgb = C_TEXT_MAIN
        pd.space_after = Pt(10)

    # Right Container: Empirical Problem Grounding
    add_card(s2, Inches(6.8), Inches(2.0), Inches(5.7), Inches(4.9))
    tb_r = s2.shapes.add_textbox(Inches(7.05), Inches(2.2), Inches(5.2), Inches(4.5))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True

    p = tf_r.paragraphs[0]
    p.text = "Benchmark Reality: The IEEE-CIS Dataset"
    p.font.bold = True
    p.font.size = Pt(14)
    p.font.color.rgb = C_TEXT_MAIN
    p.space_after = Pt(14)

    # Stat callout blocks
    stat_card1 = add_card(s2, Inches(7.05), Inches(2.8), Inches(5.2), Inches(1.0), bg_color=C_BG, border_color=C_BORDER)
    tb_sc1 = s2.shapes.add_textbox(Inches(7.2), Inches(2.88), Inches(4.9), Inches(0.85))
    tf_sc1 = tb_sc1.text_frame
    tf_sc1.word_wrap = True
    p_sc1 = tf_sc1.paragraphs[0]
    p_sc1.text = "590,540 Real-World Transactions"
    p_sc1.font.bold = True
    p_sc1.font.size = Pt(12)
    p_sc1.font.color.rgb = C_ACCENT_GREEN
    p_sc1_sub = tf_sc1.add_paragraph()
    p_sc1_sub.text = "Spanning 182 continuous calendar days of payment activity."
    p_sc1_sub.font.size = Pt(10)
    p_sc1_sub.font.color.rgb = C_TEXT_MUTED

    stat_card2 = add_card(s2, Inches(7.05), Inches(3.95), Inches(5.2), Inches(1.0), bg_color=C_BG, border_color=C_BORDER)
    tb_sc2 = s2.shapes.add_textbox(Inches(7.2), Inches(4.03), Inches(4.9), Inches(0.85))
    tf_sc2 = tb_sc2.text_frame
    tf_sc2.word_wrap = True
    p_sc2 = tf_sc2.paragraphs[0]
    p_sc2.text = "Extreme Class Imbalance: 3.499% Positive Fraud"
    p_sc2.font.bold = True
    p_sc2.font.size = Pt(12)
    p_sc2.font.color.rgb = C_ACCENT_RED
    p_sc2_sub = tf_sc2.add_paragraph()
    p_sc2_sub.text = "27.58 benign transactions for every single fraudulent transaction."
    p_sc2_sub.font.size = Pt(10)
    p_sc2_sub.font.color.rgb = C_TEXT_MUTED

    # Bottom takeaway
    tb_r_bot = s2.shapes.add_textbox(Inches(7.05), Inches(5.15), Inches(5.2), Inches(1.5))
    tf_r_bot = tb_r_bot.text_frame
    tf_r_bot.word_wrap = True
    p_rb = tf_r_bot.paragraphs[0]
    p_rb.text = "The Central Research Question:"
    p_rb.font.bold = True
    p_rb.font.size = Pt(11.5)
    p_rb.font.color.rgb = C_ACCENT_GREEN
    p_rb_t = tf_r_bot.add_paragraph()
    p_rb_t.text = "Does the substantial engineering overhead of graph neural networks and dynamic entity memory yield defensible predictive gains over modern gradient boosting under strict chronological evaluation?"
    p_rb_t.font.size = Pt(10.5)
    p_rb_t.font.color.rgb = C_TEXT_MAIN

    set_speaker_notes(s2, """[PRESENTER A - PROBLEM]
To understand our project, we must examine why transaction fraud detection is fundamentally difficult.

Traditional fraud prevention often treats transactions as isolated rows in a database. But in real financial operations, fraud analysts face three interconnected dilemmas:
1. Deceptive Camouflage: Fraudulent transactions look innocent when viewed alone—an adversary uses stolen credentials to buy everyday items at common merchants.
2. Temporal Non-Stationarity: Financial fraud is an adversarial game. As soon as a defensive rule is deployed, fraudsters pivot, causing severe model decay over time.
3. Distributed Coordination: Fraud rings systematically distribute attacks across multiple credit cards, synthetic accounts, mobile devices, and proxy IPs to stay under single-account velocity limits.

We evaluate these challenges on the standard IEEE-CIS benchmark: 590,540 real-world transactions over 182 days with severe class imbalance—only 3.499% fraud, or roughly 28 legitimate purchases for every illicit one.

This brings us to our central research inquiry: Can graph structure and temporal memory provide genuine lift, or are practitioners better served by tuned tabular gradient-boosted trees?""")

    # =========================================================================
    # SLIDE 3: METHODOLOGICAL PITFALLS & CAUSAL HYGIENE (Presenter A)
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_slide_background(s3, prs)
    add_header(s3, "PART I — FOUNDATIONS & SYSTEM DESIGN  •  PRESENTER A",
               "The Graph Illusion: Target Leakage and the Labelling Artifact",
               "Why financial graph models often appear extraordinarily strong on paper while failing in production")

    # Alert Banner Top
    add_card(s3, Inches(0.8), Inches(2.0), Inches(11.733), Inches(1.35), bg_color=RGBColor(253, 245, 244), border_color=C_ACCENT_RED)
    tb_alert = s3.shapes.add_textbox(Inches(1.05), Inches(2.12), Inches(11.2), Inches(1.1))
    tf_alert = tb_alert.text_frame
    tf_alert.word_wrap = True
    p_al_t = tf_alert.paragraphs[0]
    p_al_t.text = "THE 59.86% LABELLING ARTIFACT DISCOVERED IN IEEE-CIS"
    p_al_t.font.bold = True
    p_al_t.font.size = Pt(11.5)
    p_al_t.font.color.rgb = C_ACCENT_RED
    p_al_b = tf_alert.add_paragraph()
    p_al_b.text = "In IEEE-CIS, a single backward-looking feature—whether the same pseudo-account suffered earlier fraud—separates 59.86% fraud from 0.08%. Unconstrained graph convolutions accidentally propagate this labelling convention across neighbors or look backwards from future chargebacks, creating an illusion of high predictive power."
    p_al_b.font.size = Pt(10.5)
    p_al_b.font.color.rgb = C_TEXT_MAIN

    # Three Causal Safeguards Cards
    card_w = Inches(3.68)
    card_gap = Inches(0.34)

    # Box 1
    add_card(s3, Inches(0.8), Inches(3.55), card_w, Inches(3.35))
    tb_b1 = s3.shapes.add_textbox(Inches(0.95), Inches(3.7), card_w - Inches(0.3), Inches(3.0))
    tf_b1 = tb_b1.text_frame
    tf_b1.word_wrap = True
    pb1_h = tf_b1.paragraphs[0]
    pb1_h.text = "1. Strict Chronological Split"
    pb1_h.font.bold = True
    pb1_h.font.size = Pt(13)
    pb1_h.font.color.rgb = C_ACCENT_GREEN
    pb1_h.space_after = Pt(8)
    pb1_t = tf_b1.add_paragraph()
    pb1_t.text = "• Train: Days 1–121 (417,559 rows)\n• Validation: Days 122–151 (83,655 rows)\n• Held-Out: Days 152–182 (89,326 rows, cryptographically sealed)\n\nEncoders, imputers, and scalers are fitted strictly on Train data only. Random k-fold splits are strictly rejected."
    pb1_t.font.size = Pt(10)
    pb1_t.font.color.rgb = C_TEXT_MAIN

    # Box 2
    add_card(s3, Inches(0.8) + card_w + card_gap, Inches(3.55), card_w, Inches(3.35))
    tb_b2 = s3.shapes.add_textbox(Inches(0.95) + card_w + card_gap, Inches(3.7), card_w - Inches(0.3), Inches(3.0))
    tf_b2 = tb_b2.text_frame
    tf_b2.word_wrap = True
    pb2_h = tf_b2.paragraphs[0]
    pb2_h.text = "2. Causal Feature Masking"
    pb2_h.font.bold = True
    pb2_h.font.size = Pt(13)
    pb2_h.font.color.rgb = C_ACCENT_GREEN
    pb2_h.space_after = Pt(8)
    pb2_t = tf_b2.add_paragraph()
    pb2_t.text = "• Every feature for transaction t sees only strictly earlier rows (t' < t).\n• Handcrafted graph features use historical entity counts with zero future leakage.\n• Neural graphs score day t using graph state masked strictly to edges with timestamp <= t."
    pb2_t.font.size = Pt(10)
    pb2_t.font.color.rgb = C_TEXT_MAIN

    # Box 3
    add_card(s3, Inches(0.8) + (card_w + card_gap)*2, Inches(3.55), card_w, Inches(3.35))
    tb_b3 = s3.shapes.add_textbox(Inches(0.95) + (card_w + card_gap)*2, Inches(3.7), card_w - Inches(0.3), Inches(3.0))
    tf_b3 = tb_b3.text_frame
    tf_b3.word_wrap = True
    pb3_h = tf_b3.paragraphs[0]
    pb3_h.text = "3. Interventional Checks"
    pb3_h.font.bold = True
    pb3_h.font.size = Pt(13)
    pb3_h.font.color.rgb = C_ACCENT_GREEN
    pb3_h.space_after = Pt(8)
    pb3_t = tf_b3.add_paragraph()
    pb3_t.text = "• Interventional code tests corrupt future days and verify that earlier predictions remain bit-identical.\n• Complete separation between research evaluation and synthetic demonstration pipelines.\n• Frozen validation gate architecture prevents post-hoc hyperparameter tuning."
    pb3_t.font.size = Pt(10)
    pb3_t.font.color.rgb = C_TEXT_MAIN

    set_speaker_notes(s3, """[PRESENTER A - METHODOLOGY & CAUSALITY]
Before discussing architectures, we must confront a critical methodological pitfall that plagues financial graph literature.

Many published papers report astonishing GNN accuracy on the IEEE-CIS benchmark. But during our Phase 1 reconnaissance, we uncovered the underlying reason: a massive labelling artifact. A single feature—whether the same pseudo-account experienced fraud earlier—separates fraud at 59.86% versus 0.08%. In an unconstrained graph neural network, message-passing across transactions easily propagates chargeback labels backwards in time, effectively allowing the model to cheat.

To guarantee scientific honesty, we instituted three non-negotiable causal safeguards:
1. Strict Chronological Splitting: We evaluate only forward in time. Train covers Days 1–121, Validation covers Days 122–151, and Held-out Days 152–182 remain sealed. All feature encoders and scalers are fitted on Train only.
2. Causal Feature Masking: All tabular and graph features operate under a strict invariant: a transaction at time t can only inspect events strictly earlier than t.
3. Interventional Invariance Tests: In our test suite, we programmatically corrupt future transaction timestamps and verify that past predictions remain bit-for-bit identical.

With these safeguards established, let us examine our system architecture.""")

    # =========================================================================
    # SLIDE 4: SYSTEM ARCHITECTURE (Presenter A)
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_slide_background(s4, prs)
    add_header(s4, "PART I — FOUNDATIONS & SYSTEM DESIGN  •  PRESENTER A",
               "System Architecture: From Ingestion to Verified Intelligence",
               "A dual-pipeline design separating tabular models, relational graphs, and temporal memory")

    # Embed generated diagram
    diag_path = os.path.join(ASSETS_DIR, "diag_system_architecture.png")
    if os.path.exists(diag_path):
        s4.shapes.add_picture(diag_path, Inches(0.8), Inches(1.95), width=Inches(11.733))

    set_speaker_notes(s4, """[PRESENTER A - ARCHITECTURE]
Here is the end-to-end architecture of the software and research pipeline we implemented. It is organized into four distinct stages:

Stage 1: Ingestion and Causal Preprocessing.
Raw IEEE-CIS transaction streams are loaded, chronologically partitioned, and validated. Encoders and scalers are fitted strictly on the 121-day training set, while the held-out test period remains cryptographically locked.

Stage 2: The Dual Representation Engine.
Transactions fork into two parallel representations:
- A tabular feature space containing 470 preprocessed numerical and categorical features, plus 40 backward-looking graph features.
- A heterogeneous dynamic graph containing 5 node types and 4 timed relations.

Stage 3: The Model Evaluation Suite.
We train and evaluate:
- Standard tabular baselines (Logistic Regression, Random Forest, and tuned XGBoost).
- Comparison B: XGBoost augmented with our 40 handcrafted graph features.
- Static GNNs: Heterogeneous GraphSAGE and Graph Attention Networks with causal snapshot masking.
- Temporal GraphSAGE with continuous recurrent entity memory.

Stage 4: Intelligence and Drift Control.
Predictions feed into downstream intelligence modules: continual walk-forward adaptation to counteract drift, candidate-community discovery via Louvain clustering, exact TreeSHAP feature explanations, and a zero-dependency local serving API.

Next, let us look closely at how the heterogeneous graph was constructed.""")

    # =========================================================================
    # SLIDE 5: GRAPH SCHEMA & ENTITY REPRESENTATION (Presenter A)
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_slide_background(s5, prs)
    add_header(s5, "PART I — FOUNDATIONS & SYSTEM DESIGN  •  PRESENTER A",
               "Grounded Graph Construction: Entities and Timed Relational Edges",
               "Constructing a heterogeneous network from observed transaction fields without speculative entities")

    # Embed generated graph schema diagram
    schema_path = os.path.join(ASSETS_DIR, "diag_graph_schema.png")
    if os.path.exists(schema_path):
        s5.shapes.add_picture(schema_path, Inches(1.766), Inches(1.85), width=Inches(9.8))

    set_speaker_notes(s5, """[PRESENTER A - GRAPH SCHEMA]
A central principle of our design is that the graph schema must emerge directly from verified data properties, rather than arbitrary assumptions.

As shown in this schema diagram:
At the center is the Transaction Node, representing 590,540 events carrying 470 features and exact seconds timestamps.

Connected to transactions are four distinct entity types:
1. Pseudo-Accounts (199,625 nodes): Because real customer IDs are anonymized in IEEE-CIS, we rigorously derived a stable pseudo-account identifier using card attributes, billing region, and the D1 timedelta offset: card1 + addr1 + (day - D1). The 'charged_to' edge covers 88.9% of transactions.
2. Payment Cards (13,553 nodes): Defined by card numbers and issuers (card1 through card6), connected via the 'used_card' relation with 100% coverage.
3. Network Proxies (2,394 nodes): Infrastructure nodes derived from purchaser email domains and network proxy clusters, with 23.6% coverage via 'via_network'.
4. Device Fingerprints (4,880 nodes): Hardware and operating system combinations (id_30 through id_33) with 12.0% coverage via 'from_device'.

Crucially, in Phase 4 we measured and explicitly excluded candidates like raw billing address, email domain, and raw device model strings. These created massive artificial hubs with millions of connections but zero predictive signal above chance.

Let us now contrast handcrafted feature engineering against deep graph neural representations.""")

    # =========================================================================
    # SLIDE 6: FEATURE ENGINEERING VS LEARNED GRAPH REPS (Presenter A)
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_slide_background(s6, prs)
    add_header(s6, "PART I — FOUNDATIONS & SYSTEM DESIGN  •  PRESENTER A",
               "The Modeling Dilemma: Handcrafted Graph Features vs Deep GNNs",
               "Two competing engineering paradigms for integrating relational context into tabular streams")

    # Left Container: Comparison B (Handcrafted Graph Features)
    add_card(s6, Inches(0.8), Inches(2.0), Inches(5.7), Inches(4.9))
    tb_l = s6.shapes.add_textbox(Inches(1.05), Inches(2.2), Inches(5.2), Inches(4.5))
    tf_l = tb_l.text_frame
    tf_l.word_wrap = True

    pl_h = tf_l.paragraphs[0]
    pl_h.text = "Paradigm 1: Handcrafted Graph Features (B)"
    pl_h.font.bold = True
    pl_h.font.size = Pt(13.5)
    pl_h.font.color.rgb = C_ACCENT_GREEN
    pl_h.space_after = Pt(12)

    items_feat = [
        ("40 Label-Free Backward Features",
         "Computed per relation from strictly earlier transactions (t' < t) using running entity state."),
        ("Multi-Scale Temporal Velocity",
         "Transaction counts across prior 1-day and 7-day rolling windows for each associated entity."),
        ("Recency & Monetary Behavior",
         "Seconds elapsed since last transaction; running mean and standard deviation of log-transaction amount."),
        ("Relational Entity Diversity",
         "Count of distinct payment cards, devices, and network proxies linked to the pseudo-account to date."),
        ("Pre-Registered Protocol",
         "Feature formulas and acceptance rules locked in PREREGISTRATION_B.md before running any code.")
    ]
    for h, d in items_feat:
        ph = tf_l.add_paragraph()
        ph.text = f"• {h}: "
        ph.font.bold = True
        ph.font.size = Pt(11)
        ph.font.color.rgb = C_TEXT_MAIN
        pd = ph.add_run()
        pd.text = d
        pd.font.bold = False
        pd.font.size = Pt(10)
        pd.font.color.rgb = C_TEXT_MUTED
        ph.space_after = Pt(6)

    # Right Container: Deep Graph Neural Networks
    add_card(s6, Inches(6.8), Inches(2.0), Inches(5.7), Inches(4.9))
    tb_r = s6.shapes.add_textbox(Inches(7.05), Inches(2.2), Inches(5.2), Inches(4.5))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True

    pr_h = tf_r.paragraphs[0]
    pr_h.text = "Paradigm 2: Graph Neural Networks (C & D)"
    pr_h.font.bold = True
    pr_h.font.size = Pt(13.5)
    pr_h.font.color.rgb = C_ACCENT_GREEN
    pr_h.space_after = Pt(12)

    items_gnn = [
        ("Heterogeneous GraphSAGE (Cell C)",
         "Per-node-type linear projections with two relational SAGE aggregation layers (64 hidden units, 162k params)."),
        ("Graph Attention Networks (Cell D)",
         "Multi-head attention mechanism (4 heads, 364k params) designed to dynamically weight neighbor importance."),
        ("Causal Snapshot Masking",
         "GNNs score transactions on day t using only the graph structure formed by edges timestamped <= t."),
        ("No-Relations MLP Control",
         "An identical architecture with all relational edges removed (34k params) to isolate relational value from neural capacity."),
        ("Class-Weighted Optimization",
         "Trained with Adam (lr=1e-3) and class-weighted binary cross-entropy (weight=27.36) with early stopping.")
    ]
    for h, d in items_gnn:
        ph = tf_r.add_paragraph()
        ph.text = f"• {h}: "
        ph.font.bold = True
        ph.font.size = Pt(11)
        ph.font.color.rgb = C_TEXT_MAIN
        pd = ph.add_run()
        pd.text = d
        pd.font.bold = False
        pd.font.size = Pt(10)
        pd.font.color.rgb = C_TEXT_MUTED
        ph.space_after = Pt(6)

    set_speaker_notes(s6, """[PRESENTER A - FEATURE VS GNN]
When applying graph intelligence to tabular transactions, machine learning engineers face a fundamental fork in the road:
Do we extract explicit relational features and feed them to tree-based models, or do we train end-to-end Graph Neural Networks?

We implemented and compared both paradigms under strict pre-registered protocols.

On the left is Comparison B: Handcrafted Graph Features.
We engineered 40 label-free, backward-looking features capturing entity velocity, recency, amount volatility, and cross-entity diversity. For example, how many transactions did this pseudo-account perform in the last 24 hours versus the last 7 days? How many different credit cards or devices has it touched? These were computed strictly using past events and pre-registered before execution.

On the right are Deep Graph Neural Networks:
- Cell C: Heterogeneous GraphSAGE, utilizing two relational neighborhood aggregation layers to learn dense node representations.
- Cell D: A Graph Attention Network (GAT) with 4 attention heads to dynamically weight neighbor relevance.
- To rigorously isolate the value of graph structure, we included a No-Relations MLP Control that shares the same network capacity and tabular inputs but has all graph edges severed.

Now, let us examine how we model temporal entity memory across time.""")

    # =========================================================================
    # SLIDE 7: MODELING TEMPORAL ENTITY MEMORY (Presenter A)
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_slide_background(s7, prs)
    add_header(s7, "PART I — FOUNDATIONS & SYSTEM DESIGN  •  PRESENTER A",
               "Temporal GraphSAGE: Continuous Recurrent Entity State",
               "Preserving dynamic behavioral memory across transactions without forward leakage")

    # Embed generated temporal memory diagram
    temp_path = os.path.join(ASSETS_DIR, "diag_temporal_memory.png")
    if os.path.exists(temp_path):
        s7.shapes.add_picture(temp_path, Inches(1.8), Inches(1.82), width=Inches(9.733))

    # Bottom comparison callout
    add_card(s7, Inches(0.8), Inches(5.62), Inches(11.733), Inches(1.32), bg_color=C_CARD_BG, border_color=C_BORDER)
    tb_c = s7.shapes.add_textbox(Inches(1.05), Inches(5.72), Inches(11.2), Inches(1.1))
    tf_c = tb_c.text_frame
    tf_c.word_wrap = True
    pc_h = tf_c.paragraphs[0]
    pc_h.text = "RIGOROUS EXPERIMENTAL CONTROLS (PHASE 7A)"
    pc_h.font.bold = True
    pc_h.font.size = Pt(11)
    pc_h.font.color.rgb = C_ACCENT_GREEN
    pc_t = tf_c.add_paragraph()
    pc_t.text = "To prove that any observed temporal gain is actually caused by the recurrent GRU mechanism rather than the training regime, we evaluated two pre-registered controls: (1) Schedule-Matched Static GNN (equal optimizer steps), and (2) Memory-Ablated Temporal GNN (replacing the recurrent GRU with a simple cumulative average)."
    pc_t.font.size = Pt(9.8)
    pc_t.font.color.rgb = C_TEXT_MAIN

    set_speaker_notes(s7, """[PRESENTER A - TEMPORAL MEMORY]
In a streaming financial system, static graph snapshots miss the evolution of user behavior. A card that lay dormant for six months and suddenly fires 10 high-value transactions in an hour represents a critical behavioral signature.

To capture this, we implemented Temporal GraphSAGE with continuous recurrent entity memory.
As illustrated in the diagram:
- The system operates day-by-day.
- Each entity maintains a 64-dimensional latent state updated by a Gated Recurrent Unit (GRU).
- When transactions occur on day t, entity states are updated using the day's message snapshot and an explicit elapsed-time encoding (Delta t) representing the time since previous activity.
- The updated entity state then carries forward smoothly into the next day without reset.

Crucially, in Phase 7A we designed two controls to eliminate confounding factors:
First, a Schedule-Matched Static GNN: In standard training, temporal models take ~17 optimizer steps per epoch versus 1 for static GNNs. The schedule-matched control equalizes gradient steps so we don't mistake optimization volume for architectural superiority.
Second, a Memory-Ablated Temporal GNN: We replace the non-linear recurrent GRU cell with an unweighted cumulative mean, isolating whether recurrent memory dynamics specifically matter.

Let us now review the 5 pre-registered hypotheses before I hand over to Presenter B.""")

    # =========================================================================
    # SLIDE 8: THE MODELING STRATEGY & HYPOTHESES (Presenter A -> B Handoff)
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_slide_background(s8, prs)
    add_header(s8, "PART I — FOUNDATIONS & SYSTEM DESIGN  •  PRESENTER A",
               "Pre-Registered Research Hypotheses: What We Set Out to Test",
               "Establishing rigorous falsification criteria before executing experimental runs")

    # Left Container: Hypotheses Table
    add_card(s8, Inches(0.8), Inches(2.0), Inches(7.4), Inches(4.9))
    tb_h = s8.shapes.add_textbox(Inches(1.05), Inches(2.15), Inches(6.9), Inches(4.6))
    tf_h = tb_h.text_frame
    tf_h.word_wrap = True

    ph_t = tf_h.paragraphs[0]
    ph_t.text = "The Five Pre-Registered Research Hypotheses"
    ph_t.font.bold = True
    ph_t.font.size = Pt(13.5)
    ph_t.font.color.rgb = C_TEXT_MAIN
    ph_t.space_after = Pt(10)

    hypos = [
        ("H1: Relational Information Gain",
         "Graph structure provides predictive information beyond conventional transaction-level tabular features."),
        ("H2: Representation Superiority",
         "Learned graph neural representations (GNNs) outperform handcrafted backward-looking graph features."),
        ("H3: Temporal Memory Generalization",
         "Recurrent entity memory improves generalization to future transactions compared with static graph modeling."),
        ("H4: Continual Drift Adaptation",
         "Periodic fine-tuning on newly revealed labels significantly reduces model degradation under concept drift."),
        ("H5: Coordinated Community Discovery",
         "Graph-based community detection discovers coordinated multi-entity fraud structures that row-level classifiers miss.")
    ]
    for code, desc in hypos:
        p_c = tf_h.add_paragraph()
        p_c.text = f"• {code}: "
        p_c.font.bold = True
        p_c.font.size = Pt(11)
        p_c.font.color.rgb = C_ACCENT_GREEN
        p_d = p_c.add_run()
        p_d.text = desc
        p_d.font.bold = False
        p_d.font.size = Pt(10)
        p_d.font.color.rgb = C_TEXT_MUTED
        p_c.space_after = Pt(6)

    # Right Container: Speaker Handoff Card
    add_card(s8, Inches(8.5), Inches(2.0), Inches(4.033), Inches(4.9), bg_color=C_CARD_BG, border_color=C_ACCENT_GREEN)
    tb_ho = s8.shapes.add_textbox(Inches(8.75), Inches(2.2), Inches(3.533), Inches(4.5))
    tf_ho = tb_ho.text_frame
    tf_ho.word_wrap = True

    pho_tag = tf_ho.paragraphs[0]
    pho_tag.text = "PRESENTER HANDOFF"
    pho_tag.font.bold = True
    pho_tag.font.size = Pt(11)
    pho_tag.font.color.rgb = C_ACCENT_GREEN
    pho_tag.space_after = Pt(12)

    pho_t = tf_ho.add_paragraph()
    pho_t.text = "From System Architecture to Empirical Evidence"
    pho_t.font.bold = True
    pho_t.font.size = Pt(14)
    pho_t.font.color.rgb = C_TEXT_MAIN
    pho_t.space_after = Pt(14)

    pho_b = tf_ho.add_paragraph()
    pho_b.text = "Presenter A has established the foundational problem, causal invariants, heterogeneous graph construction, and temporal architecture.\n\nPresenter B will now present the experimental evaluation, our quantitative results across tabular and graph models, drift adaptation evidence, community discovery benchmarks, and final research conclusions."
    pho_b.font.size = Pt(10.5)
    pho_b.font.color.rgb = C_TEXT_MAIN

    set_speaker_notes(s8, """[PRESENTER A - HANDOFF TO PRESENTER B]
To hold ourselves to high scientific standards, we pre-registered five formal hypotheses—H1 through H5—with strict falsification rules before training and evaluating our models:
- H1: That graph structure adds lift over tabular features.
- H2: That deep learned graph representations beat handcrafted graph features.
- H3: That temporal entity memory improves future generalization over static models.
- H4: That adaptive updating reduces degradation under concept drift.
- H5: That graph clustering uncovers coordinated fraud rings missed by individual transaction scoring.

This concludes Part I on our problem formulation, causal hygiene, and system architecture.

I will now hand over to my co-presenter, Presenter B, who will walk you through our experimental validation, empirical findings, surprising baseline comparisons, and scientific conclusions. Presenter B, the floor is yours.""")

    # =========================================================================
    # SLIDE 9: EXPERIMENTAL PROTOCOL & VALIDATION FRAMEWORK (Presenter B)
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_slide_background(s9)
    add_header(s9, "PART II — IMPLEMENTATION & EVIDENCE  •  PRESENTER B",
               "Experimental Protocol: Metrics, Segmentation, and Integrity",
               "Why conventional accuracy metrics mislead and how we enforced verification rigor")

    # Three Vertical Columns
    col_w = Inches(3.68)
    col_gap = Inches(0.34)

    # Col 1: Metric Selection
    add_card(s9, Inches(0.8), Inches(2.0), col_w, Inches(4.9))
    tb_c1 = s9.shapes.add_textbox(Inches(1.0), Inches(2.2), col_w - Inches(0.4), Inches(4.5))
    tf_c1 = tb_c1.text_frame
    tf_c1.word_wrap = True
    p1_h = tf_c1.paragraphs[0]
    p1_h.text = "1. PR-AUC as Primary Metric"
    p1_h.font.bold = True
    p1_h.font.size = Pt(13)
    p1_h.font.color.rgb = C_ACCENT_GREEN
    p1_h.space_after = Pt(10)
    p1_t = tf_c1.add_paragraph()
    p1_t.text = "Under 27.6:1 class imbalance, standard accuracy and ROC-AUC are deeply misleading:\n\n• A naive classifier predicting 'legitimate' achieves 96.5% accuracy.\n• ROC-AUC evaluates true positive rate against false positive rate over all negatives, yielding inflated scores (> 0.85) even when most alerts are false alarms.\n• PR-AUC (Precision-Recall AUC) directly measures precision across recall thresholds and serves as our primary benchmark."
    p1_t.font.size = Pt(10.2)
    p1_t.font.color.rgb = C_TEXT_MAIN

    # Col 2: 3-Way Segmentation
    add_card(s9, Inches(0.8) + col_w + col_gap, Inches(2.0), col_w, Inches(4.9))
    tb_c2 = s9.shapes.add_textbox(Inches(1.0) + col_w + col_gap, Inches(2.2), col_w - Inches(0.4), Inches(4.5))
    tf_c2 = tb_c2.text_frame
    tf_c2.word_wrap = True
    p2_h = tf_c2.paragraphs[0]
    p2_h.text = "2. Strict 3-Way Segmentation"
    p2_h.font.bold = True
    p2_h.font.size = Pt(13)
    p2_h.font.color.rgb = C_ACCENT_GREEN
    p2_h.space_after = Pt(10)
    p2_t = tf_c2.add_paragraph()
    p2_t.text = "Aggregate metrics obscure where graph models succeed or fail. We report all results across 3 non-overlapping segments:\n\n• Warm: The pseudo-account has prior transaction history.\n• Cold-Start: Known entity, but this is its first observed transaction.\n• No-Edge: Transaction with no linked pseudo-account.\n\n*Critical Rule: Merging Cold and No-Edge artificially inflates cold-start metrics and is strictly prohibited in our protocol.*"
    p2_t.font.size = Pt(10.2)
    p2_t.font.color.rgb = C_TEXT_MAIN

    # Col 3: Hash-Recorded Table
    add_card(s9, Inches(0.8) + (col_w + col_gap)*2, Inches(2.0), col_w, Inches(4.9))
    tb_c3 = s9.shapes.add_textbox(Inches(1.0) + (col_w + col_gap)*2, Inches(2.2), col_w - Inches(0.4), Inches(4.5))
    tf_c3 = tb_c3.text_frame
    tf_c3.word_wrap = True
    p3_h = tf_c3.paragraphs[0]
    p3_h.text = "3. Verification & Sealed Test"
    p3_h.font.bold = True
    p3_h.font.size = Pt(13)
    p3_h.font.color.rgb = C_ACCENT_GREEN
    p3_h.space_after = Pt(10)
    p3_t = tf_c3.add_paragraph()
    p3_t.text = "All reported results are backed by cryptographic verification:\n\n• Unified Metric Engine: A single shared module (src/models/metrics.py) recomputed 8,184 values across 65 files with exact 0.0 discrepancy.\n• Sealed Held-Out Period: Days 152–182 remain locked under cryptographic guards. Zero held-out numbers are reported, eliminating post-hoc cherry-picking.\n• All reported metrics reflect validation-period evaluation (Days 122–151)."
    p3_t.font.size = Pt(10.2)
    p3_t.font.color.rgb = C_TEXT_MAIN

    set_speaker_notes(s9, """[PRESENTER B - EVALUATION PROTOCOL]
Thank you, Presenter A. I will now present our empirical evaluation, quantitative results, and scientific findings.

When evaluating machine learning on imbalanced fraud datasets, how you evaluate determines whether your conclusions are scientifically valid. We built our evaluation around three core principles:

1. Metric Selection: In an environment with 27.6 benign transactions for every single fraud event, accuracy is meaningless—a trivial baseline predicting all negatives gets 96.5% accuracy. Even ROC-AUC is misleadingly optimistic because a massive denominator of true negatives conceals an unmanageable flood of false alarms. We designate Precision-Recall AUC (PR-AUC) as our primary benchmark because it directly evaluates precision as a function of alert recall.

2. Strict 3-Way Entity Segmentation: Aggregating all transactions into a single metric hides where graph models actually provide value. We evaluate every model across three non-overlapping subsets: Warm transactions with established history, Cold-start transactions where an entity is seen for the first time, and No-Edge transactions. We discovered that conflating cold-start with no-edge transactions artificially inflates reported GNN performance—a practice we strictly barred.

3. Complete Metric Integrity and Sealed Test Period: To ensure absolute reproducibility, we built a consolidated verification table in Phase 12. A single metric engine recomputed all 8,184 recorded metric values with a maximum absolute discrepancy of exactly 0.0. Furthermore, Days 152–182 remain sealed under cryptographic hash guards. Every number presented today is a true validation-period finding.

Let us now examine the quantitative comparison between tabular and graph models.""")

    # =========================================================================
    # SLIDE 10: QUANTITATIVE RESULTS - TABULAR VS GRAPH (Presenter B)
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_slide_background(s10)
    add_header(s10, "PART II — IMPLEMENTATION & EVIDENCE  •  PRESENTER B",
               "The Tabular Baseline Prevails: Tuned XGBoost Outperforms Graph Models",
               "Validation-period PR-AUC results (Days 122–151) across tabular, handcrafted, and neural architectures")

    # Left: Embed Fig 1 Chart
    chart_path = os.path.join(ASSETS_DIR, "fig1_pr_auc_comparison.png")
    if os.path.exists(chart_path):
        s10.shapes.add_picture(chart_path, Inches(0.8), Inches(1.95), width=Inches(7.2))

    # Right: Summary Card of Verdicts
    add_card(s10, Inches(8.3), Inches(1.95), Inches(4.233), Inches(4.95))
    tb_v = s10.shapes.add_textbox(Inches(8.55), Inches(2.15), Inches(3.733), Inches(4.55))
    tf_v = tb_v.text_frame
    tf_v.word_wrap = True

    pv_h = tf_v.paragraphs[0]
    pv_h.text = "Key Findings & Hypothesis Verdicts"
    pv_h.font.bold = True
    pv_h.font.size = Pt(13)
    pv_h.font.color.rgb = C_TEXT_MAIN
    pv_h.space_after = Pt(10)

    verdicts = [
        ("XGBoost Baseline Strongest",
         "Tuned XGBoost on tabular features achieves 0.5395 PR-AUC (weighted) and 0.5948 (unweighted), outperforming all graph models."),
        ("Comparison B (XGB + 40 Graph Feat.)",
         "Primary recipe reaches 0.5440 (+0.0046). Across 3 subsampled seeds, delta is -0.0005 vs SD 0.0034. H1-direct is NOT ESTABLISHED."),
        ("H2 Contradicted Under Rule",
         "Handcrafted graph features (0.5611) outperform Temporal GraphSAGE (0.4927) and GraphSAGE (0.4637). Learned embeddings did not beat handcrafted features."),
        ("GAT Convergence Failure",
         "GAT hit the 150-epoch ceiling at 0.3320 PR-AUC without converging. We report this transparently as an under-trained failure, not an inherent flaw.")
    ]
    for h, d in verdicts:
        p = tf_v.add_paragraph()
        p.text = f"• {h}: "
        p.font.bold = True
        p.font.size = Pt(10.5)
        p.font.color.rgb = C_ACCENT_GREEN if "XGBoost" in h or "Comparison" in h else C_ACCENT_RED
        pr = p.add_run()
        pr.text = d
        pr.font.bold = False
        pr.font.size = Pt(9.5)
        pr.font.color.rgb = C_TEXT_MAIN
        p.space_after = Pt(6)

    set_speaker_notes(s10, """[PRESENTER B - RESULTS TABULAR VS GRAPH]
Here are our primary empirical findings on the validation period. This analytical chart presents the core quantitative comparison across all models.

The primary takeaway is striking: Tuned tabular XGBoost is extraordinarily difficult to beat.
- Standard XGBoost on transaction features achieves a PR-AUC of 0.5395 with class weighting, and 0.5948 without class weighting.
- Static GraphSAGE scores 0.4637. While it beats its no-relations MLP control (0.4167), it remains far below the tabular model.
- Temporal GraphSAGE achieves 0.4927 across three seeds. This is the strongest graph model in our study, yet it still trails baseline XGBoost by nearly 0.047 PR-AUC.
- The Graph Attention Network (GAT) never converged within the 150-epoch limit, achieving only 0.3320. We report this openly as a negative result rather than quietly discarding it.

Now let us examine Comparison B—adding 40 handcrafted backward-looking graph features to XGBoost:
- In the primary single-run comparison, it shows a modest gain of +0.0046 (0.5440 vs 0.5395).
- However, when evaluated over three subsampled seeds, the mean difference is -0.0005 against a seed standard deviation of 0.0034.
- Under our pre-registered decision rule, H1-direct is NOT ESTABLISHED.
- Furthermore, H2 is CONTRADICTED: learned neural graph embeddings do not outperform handcrafted features; both neural models fall far below the handcrafted tabular model.

Where, then, does relational and temporal information actually help? Let us examine the ablation studies.""")

    # =========================================================================
    # SLIDE 11: TEMPORAL MEMORY GAINS & RELATION ABLATIONS (Presenter B)
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_slide_background(s11)
    add_header(s11, "PART II — IMPLEMENTATION & EVIDENCE  •  PRESENTER B",
               "Temporal Memory Yields Modest, Cold-Start Concentrated Gains",
               "Controlled ablation studies reveal where relational dynamics and memory actually matter")

    # Left Container: Recurrent Entity Memory (H3)
    add_card(s11, Inches(0.8), Inches(2.0), Inches(5.7), Inches(4.9))
    tb_l = s11.shapes.add_textbox(Inches(1.05), Inches(2.2), Inches(5.2), Inches(4.5))
    tf_l = tb_l.text_frame
    tf_l.word_wrap = True

    pl_h = tf_l.paragraphs[0]
    pl_h.text = "Temporal Memory Evaluation (H3)"
    pl_h.font.bold = True
    pl_h.font.size = Pt(13.5)
    pl_h.font.color.rgb = C_ACCENT_GREEN
    pl_h.space_after = Pt(10)

    items_mem = [
        ("Beat Memory-Ablated Control in 3 of 3 Seeds",
         "Temporal GraphSAGE (0.4927) outperforms the cumulative-mean control (0.4790) by +0.0137 pooled PR-AUC, exceeding the seed standard deviation of both arms."),
        ("Beat Schedule-Matched Static GNN in 2 of 3 Seeds",
         "Temporal model beats static GraphSAGE (0.4792) by +0.0135 on average, confirming that the gain is not an artifact of training schedule."),
        ("Gain Heavily Concentrated in Cold-Start",
         "On cold-start transactions (first transaction for a known entity), recurrent memory provides a notable +0.027 PR-AUC lift (0.3562 vs 0.3288)."),
        ("Verdict on H3: Partially Supported",
         "Recurrent memory demonstrates a modest, statistically detectable benefit that is concentrated on early entity interactions.")
    ]
    for h, d in items_mem:
        p = tf_l.add_paragraph()
        p.text = f"• {h}: "
        p.font.bold = True
        p.font.size = Pt(11)
        p.font.color.rgb = C_TEXT_MAIN
        pr = p.add_run()
        pr.text = d
        pr.font.bold = False
        pr.font.size = Pt(10)
        pr.font.color.rgb = C_TEXT_MUTED
        p.space_after = Pt(6)

    # Right Container: Relation-Level Ablation Study
    add_card(s11, Inches(6.8), Inches(2.0), Inches(5.7), Inches(4.9))
    tb_r = s11.shapes.add_textbox(Inches(7.05), Inches(2.2), Inches(5.2), Inches(4.5))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True

    pr_h = tf_r.paragraphs[0]
    pr_h.text = "Relation Ablation Study (Temporal GraphSAGE)"
    pr_h.font.bold = True
    pr_h.font.size = Pt(13.5)
    pr_h.font.color.rgb = C_ACCENT_GREEN
    pr_h.space_after = Pt(10)

    items_abl = [
        ("Pseudo-Account ('charged_to') Is Critical",
         "Removing the pseudo-account relation drops pooled PR-AUC by -0.0309 across all 3 seeds (0.4618 vs 0.4927). In the warm segment, PR-AUC collapses from 0.3213 to 0.2290."),
        ("Device Fingerprint ('from_device') Contributes",
         "Removing device edges drops PR-AUC by -0.0073 in 3 of 3 seeds (0.4854 vs 0.4927), a modest contribution slightly above rerun noise."),
        ("Card ('used_card') & Network ('via_network') Non-Essential",
         "Removing used_card (-0.0038) or via_network (-0.0053) changes PR-AUC within our 0.0058 run-to-run noise floor. Removing them does not hurt performance."),
        ("Key Structural Takeaway",
         "Nearly all relational signal exploited by the temporal model flows through the derived pseudo-account identity key.")
    ]
    for h, d in items_abl:
        p = tf_r.add_paragraph()
        p.text = f"• {h}: "
        p.font.bold = True
        p.font.size = Pt(11)
        p.font.color.rgb = C_TEXT_MAIN
        pr = p.add_run()
        pr.text = d
        pr.font.bold = False
        pr.font.size = Pt(10)
        pr.font.color.rgb = C_TEXT_MUTED
        p.space_after = Pt(6)

    set_speaker_notes(s11, """[PRESENTER B - TEMPORAL MEMORY & ABLATIONS]
While neural graph models did not surpass tabular baselines overall, our ablation experiments isolate exactly where relational structure and memory provide genuine lift.

First, evaluating Hypothesis H3 (Temporal Memory):
When compared against our memory-ablated control—which uses an identical architecture with a cumulative mean—Temporal GraphSAGE wins in 3 of 3 seeds with a mean pooled gain of +0.0137 PR-AUC.
When we inspect our 3-way segmentation, the origin of this gain becomes clear: on cold-start transactions, temporal memory provides a substantial +0.027 PR-AUC lift (0.3562 vs 0.3288). When a previously unseen card arrives on a known account, the accumulated GRU state provides critical predictive context. Therefore, H3 is partially supported with a modest, targeted effect.

Second, our pre-registered relation-level ablation study on Temporal GraphSAGE:
We systematically retrained the temporal model across three seeds with individual relations excised:
- Excising the 'charged_to' pseudo-account relation causes a severe drop of -0.0309 PR-AUC across all seeds (falling from 0.4927 to 0.4618), with warm-segment performance plummeting.
- Excising the 'from_device' relation causes a modest drop of -0.0073 PR-AUC.
- In contrast, excising payment card edges (-0.0038) or network proxy edges (-0.0053) produces differences entirely within our measured run-to-run retraining noise floor of 0.0058.

This demonstrates that almost all relational value in the graph model is concentrated in the derived pseudo-account identity key.

Now let us turn to another central finding: concept drift and continual adaptation.""")

    # =========================================================================
    # SLIDE 12: DRIFT & CONTINUAL ADAPTIVE UPDATING (Presenter B)
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    add_slide_background(s12)
    add_header(s12, "PART II — IMPLEMENTATION & EVIDENCE  •  PRESENTER B",
               "Adapting Under Drift: Periodic Fine-Tuning Outperforms Static Models",
               "Evaluating walk-forward adaptation across five consecutive 6-day validation windows")

    # Left: Embed Fig 2 Chart
    chart2_path = os.path.join(ASSETS_DIR, "fig2_drift_adaptation.png")
    if os.path.exists(chart2_path):
        s12.shapes.add_picture(chart2_path, Inches(0.8), Inches(1.95), width=Inches(7.2))

    # Right: Summary Card of Drift Findings
    add_card(s12, Inches(8.3), Inches(1.95), Inches(4.233), Inches(4.95))
    tb_d = s12.shapes.add_textbox(Inches(8.55), Inches(2.15), Inches(3.733), Inches(4.55))
    tf_d = tb_d.text_frame
    tf_d.word_wrap = True

    pd_h = tf_d.paragraphs[0]
    pd_h.text = "Drift Dynamics & H4 Verdict"
    pd_h.font.bold = True
    pd_h.font.size = Pt(13)
    pd_h.font.color.rgb = C_TEXT_MAIN
    pd_h.space_after = Pt(10)

    drift_items = [
        ("Severe Performance Decay",
         "Frozen model PR-AUC collapses from 0.5868 in Window 1 to 0.3350 in Window 5 (a drop of > 0.25) over just 30 calendar days."),
        ("Mild Covariate Shift",
         "Feature PSI across 470 features averages only ~0.01. Input distributions barely moved, yet predictive efficacy decayed sharply."),
        ("H4 Supported Under Locked Rule",
         "Adaptive arm (0.4580) degrades significantly less than Frozen (0.4243) and beats Replay Control (0.4179) in 3 of 3 seeds (+0.0402 margin)."),
        ("Replay Control Proves Cause",
         "Replay control resampled old training data with equal compute. It scored below Frozen, proving that recent labels—not compute volume—drive the gain."),
        ("Triggered Refresh Failure",
         "A score-distribution PSI trigger never fired (PSI <= 0.025 vs 0.10 threshold) while accuracy plunged. Label-free output drift failed to signal decay.")
    ]
    for h, d in drift_items:
        p = tf_d.add_paragraph()
        p.text = f"• {h}: "
        p.font.bold = True
        p.font.size = Pt(10.5)
        p.font.color.rgb = C_ACCENT_GREEN if "H4" in h or "Replay" in h else C_TEXT_MAIN
        pr = p.add_run()
        pr.text = d
        pr.font.bold = False
        pr.font.size = Pt(9.5)
        pr.font.color.rgb = C_TEXT_MUTED
        p.space_after = Pt(6)

    set_speaker_notes(s12, """[PRESENTER B - DRIFT & ADAPTATION]
One of the most consequential contributions of our research is Phase 8: our controlled study of concept drift and walk-forward adaptation over the 30-day validation period.

We divided the validation period into five consecutive 6-day windows and tracked three arms initialized from the same day-111 checkpoint:
1. Frozen baseline: Parameters locked at day 111.
2. Continual Adaptive Fine-Tuning: Model warm-start fine-tuned after each window using freshly revealed validation labels.
3. Replay Control: Spends the identical compute budget fine-tuning exclusively on resampled historical training days.

Look at the trajectory in Figure 2:
The frozen model suffers severe degradation—its PR-AUC plummets from 0.5868 in Window 1 down to 0.3350 in Window 5, a decay of more than 0.25 points in 30 days!
Crucially, when we measured covariate shift across all 470 input features, the mean Population Stability Index (PSI) was only 0.01. Raw input distributions barely shifted, yet predictive performance degraded precipitously.

Our Continual Adaptive arm maintains substantially higher performance, finishing at 0.3798 with a window-mean PR-AUC of 0.4580 versus 0.4243 for the frozen model.
Importantly, our Replay Control scored 0.4179—below the frozen baseline! This proves that the adaptive gain does not stem from extra training steps or compute volume, but strictly from adapting to recent adversarial label distributions.
Under our pre-registered decision rules, Hypothesis H4 is fully SUPPORTED.

However, we also report a notable negative finding: a pre-registered drift-triggered refresh based on the PSI of the prediction score distribution never fired (PSI stayed below 0.025). Output distribution shift failed to signal label-conditional degradation.

Next, let us examine candidate-community discovery and explainability.""")

    # =========================================================================
    # SLIDE 13: CANDIDATE-COMMUNITY DISCOVERY & EXPLAINABILITY (Presenter B)
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    add_slide_background(s13)
    add_header(s13, "PART II — IMPLEMENTATION & EVIDENCE  •  PRESENTER B",
               "Coordinated Community Discovery and Transparent Explanations",
               "Uncovering multi-entity structures and generating faithful model attributions")

    # Left Container: Candidate Communities (H5)
    add_card(s13, Inches(0.8), Inches(2.0), Inches(5.7), Inches(4.9))
    tb_l = s13.shapes.add_textbox(Inches(1.05), Inches(2.2), Inches(5.2), Inches(4.5))
    tf_l = tb_l.text_frame
    tf_l.word_wrap = True

    pl_h = tf_l.paragraphs[0]
    pl_h.text = "Candidate Community Discovery (H5)"
    pl_h.font.bold = True
    pl_h.font.size = Pt(13.5)
    pl_h.font.color.rgb = C_ACCENT_GREEN
    pl_h.space_after = Pt(10)

    items_comm = [
        ("Bipartite Louvain Modularity Clustering",
         "Constructed dynamic bipartite graphs linking card tokens to shared infrastructure (devices, netproxies) over 7-day rolling windows."),
        ("Original Synthetic Benchmark Failed Gate G4",
         "In Phase 9, the original synthetic benchmark failed frozen interference gate G4 on 2 of 3 seeds. Following scientific integrity rules, H5 was declared NOT ESTABLISHED."),
        ("Phase 9b Pre-Registered Benchmark Passed",
         "A separately pre-registered benchmark with corrected interference ranking passed all gates. The structure-plus-time method recovered camouflaged planted groups (AUPRC 0.14–0.27 vs <0.05 baselines)."),
        ("Essential Terminology Boundary",
         "Real-data clusters are strictly designated 'candidate coordinated communities'. Because IEEE-CIS contains zero verified fraud ring ground truth, claiming confirmed fraud rings is scientifically false.")
    ]
    for h, d in items_comm:
        p = tf_l.add_paragraph()
        p.text = f"• {h}: "
        p.font.bold = True
        p.font.size = Pt(11)
        p.font.color.rgb = C_TEXT_MAIN
        pr = p.add_run()
        pr.text = d
        pr.font.bold = False
        pr.font.size = Pt(10)
        pr.font.color.rgb = C_TEXT_MUTED
        p.space_after = Pt(6)

    # Right Container: Explainability Mechanisms
    add_card(s13, Inches(6.8), Inches(2.0), Inches(5.7), Inches(4.9))
    tb_r = s13.shapes.add_textbox(Inches(7.05), Inches(2.2), Inches(5.2), Inches(4.5))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True

    pr_h = tf_r.paragraphs[0]
    pr_h.text = "Transparent Explanations & Evidence"
    pr_h.font.bold = True
    pr_h.font.size = Pt(13.5)
    pr_h.font.color.rgb = C_ACCENT_GREEN
    pr_h.space_after = Pt(10)

    items_exp = [
        ("Exact TreeSHAP Feature Attributions",
         "For XGBoost, native TreeSHAP feature contributions sum exactly to the model margin (verified error < 8.1e-06 across 300 pre-registered validation rows)."),
        ("Relational Feature Influence in Model B",
         "In Comparison B, graph features appear in the top-10 contributions for 55.3% of transactions (40.0% of top-scoring alerts and 70.7% of uniform draws)."),
        ("Counterfactual Link Removals in GNNs",
         "For Temporal GraphSAGE, link removal perturbations confirmed pseudo-account edge removal changed output logit by 3.158 on average."),
        ("Attribution vs Causality Disclaimer",
         "SHAP values and link sensitivities faithfully describe model margin sensitivity, but do not establish real-world real-time causal guilt.")
    ]
    for h, d in items_exp:
        p = tf_r.add_paragraph()
        p.text = f"• {h}: "
        p.font.bold = True
        p.font.size = Pt(11)
        p.font.color.rgb = C_TEXT_MAIN
        pr = p.add_run()
        pr.text = d
        pr.font.bold = False
        pr.font.size = Pt(10)
        pr.font.color.rgb = C_TEXT_MUTED
        p.space_after = Pt(6)

    set_speaker_notes(s13, """[PRESENTER B - COMMUNITIES & EXPLAINABILITY]
In Phase 9 and Phase 11, we investigated coordinated multi-entity fraud discovery and transparent model explainability.

First, Candidate Communities:
To uncover coordinated groups, we applied Louvain modularity clustering on bipartite card-infrastructure graphs over 7-day windows.
Here we demonstrate another example of scientific transparency:
In our original Phase 9 benchmark, the synthetic test failed a frozen validity gate (Gate G4) because real transaction ranks shifted. Rather than quietly modifying the gate post-hoc, we adhered to our pre-registration: Hypothesis H5 was marked NOT ESTABLISHED.
Subsequently, we pre-registered Phase 9b with an amended interference formula. Phase 9b passed all gates and verified that structure-plus-time recovers camouflaged planted groups with AUPRC 0.14 to 0.27, outperforming baselines (< 0.05).
However, on real IEEE-CIS data, we strictly designate clusters as 'candidate coordinated communities'—never 'fraud rings'—because IEEE-CIS contains zero verified fraud ring ground truth.

Second, Explainability:
Black-box risk scores are unacceptable in financial forensics. We implemented:
- Exact TreeSHAP feature contributions for XGBoost, verified to sum to the model margin with an absolute discrepancy under 8.1e-6.
- In Comparison B, graph features ranked among the top-10 contributors in 55.3% of evaluated validation transactions.
- For Temporal GraphSAGE, counterfactual edge removals demonstrated that excising the pseudo-account edge changed model logits by 3.16 on average.
We explicitly communicate that feature attributions describe mathematical model sensitivity, not legal causality.

Let us now look at an unexpected finding on class imbalance, and our production software implementation.""")

    # =========================================================================
    # SLIDE 14: CLASS IMBALANCE INSIGHTS & SOFTWARE STACK (Presenter B)
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    add_slide_background(s14)
    add_header(s14, "PART II — IMPLEMENTATION & EVIDENCE  •  PRESENTER B",
               "The Class Weighting Penalty and Production-Grade Software",
               "Surprising loss dynamics and an un-mocked standard-library serving stack")

    # Left: Embed Fig 3 Chart
    chart3_path = os.path.join(ASSETS_DIR, "fig3_class_weighting.png")
    if os.path.exists(chart3_path):
        s14.shapes.add_picture(chart3_path, Inches(0.8), Inches(1.95), width=Inches(6.5))

    # Right: Production Software Stack Card
    add_card(s14, Inches(7.6), Inches(1.95), Inches(4.933), Inches(4.95))
    tb_s = s14.shapes.add_textbox(Inches(7.85), Inches(2.15), Inches(4.433), Inches(4.55))
    tf_s = tb_s.text_frame
    tf_s.word_wrap = True

    ps_h = tf_s.paragraphs[0]
    ps_h.text = "Production Software Implementation"
    ps_h.font.bold = True
    ps_h.font.size = Pt(13)
    ps_h.font.color.rgb = C_TEXT_MAIN
    ps_h.space_after = Pt(10)

    software_items = [
        ("The Class Imbalance Discovery",
         "Common practice applies full class weighting (scale_pos_weight = 27.36). Unweighted XGBoost (0.5948) beats full weighting (0.5395) by +0.0553 PR-AUC! Weighting inflates false alarms at high score thresholds—a swing larger than any graph enhancement."),
        ("Zero-Dependency Local Serving API",
         "Built entirely using Python's standard library (http.server, socketserver). Serves recorded validation tables, aggregate cards, and live scoring endpoints without Flask, FastAPI, or Node."),
        ("Working End-to-End Synthetic Demo",
         "scripts/demo_end_to_end.py runs on any machine with zero data prerequisites. Executes causal feature extraction, XGBoost scoring, and Louvain community discovery with byte-identical output."),
        ("381 Automated Tests with Mutation Testing",
         "Full test suite verifying leakage invariants, serving contracts, and error responses. Mutation testing confirmed 15 of 17 test mutants detected.")
    ]
    for h, d in software_items:
        p = tf_s.add_paragraph()
        p.text = f"• {h}: "
        p.font.bold = True
        p.font.size = Pt(10.5)
        p.font.color.rgb = C_ACCENT_GREEN if "Discovery" not in h else C_ACCENT_RED
        pr = p.add_run()
        pr.text = d
        pr.font.bold = False
        pr.font.size = Pt(9.5)
        pr.font.color.rgb = C_TEXT_MUTED
        p.space_after = Pt(6)

    set_speaker_notes(s14, """[PRESENTER B - CLASS IMBALANCE & SOFTWARE]
Slide 14 presents two vital aspects of our project: an unexpected machine learning insight on class imbalance, and our production software architecture.

On the left is our class imbalance study:
Conventional machine learning advice dictates that when faced with 27.6:1 class imbalance, one should set scale_pos_weight = 27.36 to equalize positive and negative loss gradients.
However, our pre-registered check revealed a stunning result:
Unweighted XGBoost achieves a PR-AUC of 0.5948, outperforming fully weighted XGBoost (0.5395) by +0.0553 points!
Why? Full class weighting forces the tree split criteria to aggressively prioritize recall, which severely inflates false-positive alerts at operating thresholds and heavily penalizes precision.
In fact, this class weighting decision caused a larger performance swing (+0.0553) than any graph feature (+0.0046) or neural network architecture in our entire study.

On the right is our software engineering implementation:
This repository is not a collection of disconnected Jupyter notebooks; it is a tested, production-grade system:
- In Phase 13, we built an interactive investigation dashboard and REST API using Python's standard library alone—no Flask, FastAPI, or external web frameworks.
- We developed an end-to-end demonstration script (demo_end_to_end.py) that generates synthetic transaction streams, computes backward graph features, executes scoring, and detects planted communities on any computer without needing Kaggle credentials.
- The repository contains 381 automated tests, including mutation sensitivity tests where 15 of 17 injected bugs were detected.

Let us now conclude with our key takeaways and honest limitations.""")

    # =========================================================================
    # SLIDE 15: CONCLUSIONS & SCIENTIFIC TAKEAWAYS (Presenter B)
    # =========================================================================
    s15 = prs.slides.add_slide(blank_layout)
    add_slide_background(s15)
    add_header(s15, "PART II — IMPLEMENTATION & EVIDENCE  •  PRESENTER B",
               "Conclusions: Lessons for Financial Graph Machine Learning",
               "Three core takeaways, acknowledged limitations, and future research directions")

    # Left Container: Three Core Takeaways
    add_card(s15, Inches(0.8), Inches(2.0), Inches(5.7), Inches(4.9))
    tb_l = s15.shapes.add_textbox(Inches(1.05), Inches(2.2), Inches(5.2), Inches(4.5))
    tf_l = tb_l.text_frame
    tf_l.word_wrap = True

    pl_h = tf_l.paragraphs[0]
    pl_h.text = "Three Core Scientific Takeaways"
    pl_h.font.bold = True
    pl_h.font.size = Pt(13.5)
    pl_h.font.color.rgb = C_ACCENT_GREEN
    pl_h.space_after = Pt(12)

    takeaways = [
        ("1. The Enduring Power of Tabular Baselines",
         "A tuned gradient-boosted tree on transaction features remains exceptionally difficult to beat. Relational graph intelligence must be evaluated against strong tabular baselines under rigorous causal controls."),
        ("2. Specificity of Relational & Temporal Value",
         "Graph and temporal dynamics provide genuine value, but in targeted niches: recurrent memory delivers meaningful lift (+0.027 PR-AUC) in cold-start scenarios, and periodic adaptation successfully counters drift."),
        ("3. Scientific Hygiene Over Sensation",
         "Negative findings, failed benchmark gates, and sealed held-out periods are essential research contributions that prevent the reproducibility crises prevalent in applied financial ML.")
    ]
    for h, d in takeaways:
        p = tf_l.add_paragraph()
        p.text = h
        p.font.bold = True
        p.font.size = Pt(11.5)
        p.font.color.rgb = C_TEXT_MAIN
        pd = tf_l.add_paragraph()
        pd.text = d
        pd.font.size = Pt(10)
        pd.font.color.rgb = C_TEXT_MUTED
        pd.space_after = Pt(10)

    # Right Container: Limitations & Future Directions
    add_card(s15, Inches(6.8), Inches(2.0), Inches(5.7), Inches(4.9))
    tb_r = s15.shapes.add_textbox(Inches(7.05), Inches(2.2), Inches(5.2), Inches(4.5))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True

    pr_h = tf_r.paragraphs[0]
    pr_h.text = "Honest Limitations & Next Steps"
    pr_h.font.bold = True
    pr_h.font.size = Pt(13.5)
    pr_h.font.color.rgb = C_TEXT_MAIN
    pr_h.space_after = Pt(12)

    limits = [
        ("Validation-Period Scope",
         "All empirical claims reflect the validation period (Days 122–151). The held-out period (Days 152–182) remains cryptographically sealed awaiting authorization."),
        ("GNN Batch Latency vs Streaming Serving",
         "Full-batch GNNs require substantial GPU memory and whole-day graph state, making them impractical for single-row sub-100ms real-time payment authorization."),
        ("Candidate Communities vs Fraud Rings",
         "Unsupervised graph clusters reflect behavioral similarity; formal operational fraud-ring confirmation requires forensic ground truth unavailable in public data."),
        ("Future Work: Streaming Graph Stores",
         "Developing incremental dynamic graph representations that maintain cold-start entity memory within millisecond production latencies.")
    ]
    for h, d in limits:
        p = tf_r.add_paragraph()
        p.text = f"• {h}: "
        p.font.bold = True
        p.font.size = Pt(10.5)
        p.font.color.rgb = C_ACCENT_GREEN if "Future" in h else C_TEXT_MAIN
        pr = p.add_run()
        pr.text = d
        pr.font.bold = False
        pr.font.size = Pt(10)
        pr.font.color.rgb = C_TEXT_MUTED
        p.space_after = Pt(8)

    set_speaker_notes(s15, """[PRESENTER B - CONCLUSIONS & CLOSING]
To conclude our presentation, we leave you with three memorable scientific takeaways:

1. The Enduring Power of Tabular Baselines:
In financial transaction fraud detection, a properly tuned gradient-boosted tree on raw transaction features remains a formidable benchmark. Many reported GNN gains in the literature vanish once strict causal boundaries and backward-looking leakage protections are enforced.

2. Relational Value is Highly Specific, Not Blanket:
Graph and temporal dynamics do contribute real value, but in distinct operational niches:
- Continuous recurrent entity memory provides substantial lift (+0.027 PR-AUC) for cold-start transactions where historical account context is otherwise absent.
- Continual walk-forward fine-tuning successfully arrests the severe model decay caused by adversarial concept drift.

3. Scientific Hygiene Protects Real-World Systems:
Reporting negative results—such as our failed Phase 9 benchmark gate, GAT convergence limits, and the unestablished direct lift of 40 graph features—is vital for sound engineering.

Our work has clear, acknowledged boundaries: all numbers reflect the validation period, GNN batch computation remains challenging for sub-100ms streaming inference, and candidate communities require forensic external confirmation.

Thank you very much for your time and attention. Both Presenter A and I are now ready to take your questions.""")

    prs.save(PPTX_PATH)
    print(f"Presentation successfully saved to: {PPTX_PATH}")


if __name__ == "__main__":
    build_deck()
