# Master Presenter Guide: Adaptive Temporal Graph Intelligence for Fraud Detection

**Presentation Deck:** `presentation/Adaptive_Temporal_Fraud_Detection.pptx`  
**Target Duration:** 20–25 minutes (plus 10–15 minutes Q&A)  
**Structure:** 2 Presenters • 15 Widescreen (16:9) Slides • 2 Thematic Parts  
**Standing Scientific Scope:** All empirical findings reflect the **validation period** (Days 122–151 of IEEE-CIS). The **held-out period** (Days 152–182) remains cryptographically sealed awaiting authorization. Real-data clusters are strictly designated *candidate coordinated communities*, not confirmed fraud rings.

---

## Presentation Delivery & Division of Responsibility

| Section | Slides | Presenter | Core Narrative Focus |
|---|---|---|---|
| **PART I: FOUNDATIONS & SYSTEM DESIGN** | Slides 1–8 | **Presenter A** | Problem formulation, adversarial camouflage, the 59.86% labelling artifact, causal leakage safeguards, 4-stage system architecture, grounded heterogeneous graph construction, handcrafted vs. deep representations, and continuous recurrent entity memory. |
| **PART II: IMPLEMENTATION & EVIDENCE** | Slides 9–15 | **Presenter B** | Metric selection (PR-AUC vs. ROC-AUC), 3-way entity segmentation, tabular vs. graph empirical findings, temporal memory cold-start lift, relation ablations, 5-window walk-forward drift adaptation, candidate-community discovery, explainability, class weighting penalty, and research conclusions. |

---

# PART I — FOUNDATIONS & SYSTEM DESIGN (PRESENTER A)

### Slide 1: Title Slide (Editorial Financial Systems Research Lab Cover)
- **Speaking Script:**
  > "Good morning, members of the evaluation committee. Today, my co-presenter and I are presenting our research project: *Adaptive Temporal Graph Intelligence for Fraud Detection and Coordinated Fraud Discovery*.
  >
  > In modern financial networks, fraud detection is increasingly framed as a graph problem. However, our study investigates a critical and often overlooked question: Does modeling complex relational graphs and temporal behavioral memory actually yield defensible predictive gains over mature, highly tuned tabular baselines like XGBoost when evaluated under strict real-world temporal causality and adversarial drift?
  >
  > To ensure a structured presentation, we have divided the talk into two balanced sections. I will deliver Part I, covering our problem formulation, causal safeguards, end-to-end architecture, heterogeneous graph construction, and temporal memory modeling. My co-presenter will then deliver Part II, covering our experimental validation, quantitative results, drift adaptation findings, explainability modules, and scientific conclusions.
  >
  > Let us begin with the core operational challenge of financial fraud."
- **Slide Takeaway:** Establish scientific rigor, operational context, and the two-presenter structure.
- **Transition:** "Let us look at why transaction fraud is fundamentally difficult for conventional machine learning."

---

### Slide 2: The Core Challenge — Fraud in High-Velocity Transaction Streams
- **Speaking Script:**
  > "Traditional fraud detection systems frequently treat transactions as independent, identically distributed events in a flat tabular database. However, in real financial streams, security analysts face three fundamental adversarial dilemmas:
  >
  > 1. Deceptive Camouflage: Fraudulent transactions do not arrive with obvious anomalies. Bad actors deliberately mimic legitimate cardholder spending in purchase amounts, merchant categories, and transaction hours.
  > 2. Temporal Non-Stationarity: Financial fraud is an evolving adversarial game. The moment a fraud prevention rule is deployed, attackers shift their tactics, causing static models to experience rapid performance decay within weeks.
  > 3. Distributed Coordination: Professional fraud syndicates do not attack using a single account. They distribute illicit velocity across dozens of synthetic accounts, credit cards, device fingerprints, and network proxies to stay beneath single-entity velocity thresholds.
  >
  > We evaluate these challenges on the real-world IEEE-CIS benchmark: 590,540 transactions spanning 182 continuous calendar days. The dataset exhibits extreme class imbalance: only 3.499% of transactions are fraudulent—roughly 28 benign purchases for every single fraud event.
  >
  > This leads to our motivating inquiry: Can relational graph structures and continuous temporal memory justify their substantial computational overhead compared to strong gradient-boosted trees?"
- **Slide Takeaway:** Financial fraud is camouflaged, non-stationary, and coordinated under extreme class imbalance.
- **Transition:** "Before discussing architectures, we must address a critical methodological vulnerability that invalidates much of the published literature on this benchmark."

---

### Slide 3: Methodological Pitfalls & The Need for Causal Hygiene
- **Speaking Script:**
  > "In the academic literature, many papers claim massive performance gains using Graph Neural Networks on IEEE-CIS. However, our Phase 1 reconnaissance revealed why many of these claims fail in production: a severe target-leakage artifact.
  >
  > In IEEE-CIS, a single backward-looking flag—whether the same pseudo-account suffered earlier fraud—separates fraud with an astonishing 59.86% prevalence compared to 0.08% for accounts without prior fraud. When an unconstrained Graph Neural Network performs neighborhood aggregation across transactions, it inadvertently propagates these chargeback labels backwards in time, effectively allowing the model to peek into the future and cheat.
  >
  > To guarantee absolute scientific integrity, we implemented three non-negotiable causal safeguards:
  > 1. Strict Chronological Splitting: We evaluate only forward in time. Days 1 to 121 form our training set; Days 122 to 151 form our validation set; and Days 152 to 182 remain cryptographically sealed as our held-out test set. All encoders and scalers are fitted strictly on training days.
  > 2. Causal Feature Masking: Every tabular and graph feature is strictly backward-looking ($t' < t$). Graph neural networks evaluate Day $t$ using graph state masked strictly to edges with timestamps at or before Day $t$.
  > 3. Interventional Invariance Checks: Our automated test suite programmatically corrupts future transaction rows and confirms that historical predictions remain bit-for-bit identical.
  >
  > With these safeguards established, let us examine our system architecture."
- **Slide Takeaway:** The 59.86% account-level labelling artifact causes spurious GNN gains if temporal leakage is not rigorously eliminated.
- **Transition:** "Here is how our complete software and research pipeline is architected."

---

### Slide 4: End-to-End System Architecture
- **Speaking Script:**
  > "This diagram illustrates our end-to-end architecture, organized into four decoupled stages:
  >
  > Stage 1: Ingestion and Causal Preprocessing. Raw transaction and identity streams are loaded, chronologically partitioned, and validated. Median imputers and frequency encoders are fitted on training data only. The held-out period remains sealed under cryptographic guards.
  >
  > Stage 2: The Dual Representation Engine. Transactions fork into two parallel representations: a 470-dimensional tabular feature space plus 40 backward-looking graph features, alongside a dynamic heterogeneous graph with 5 node types and 4 timed relations.
  >
  > Stage 3: The Model Evaluation Suite. We train and benchmark tabular baselines (Logistic Regression, Random Forest, tuned XGBoost), our pre-registered Comparison B (XGBoost with 40 graph features), static GNNs (HeteroSAGE and GAT), and Temporal GraphSAGE with continuous recurrent entity memory.
  >
  > Stage 4: Intelligence and Drift Control. Model outputs feed into downstream operational modules: continual walk-forward adaptation to counter drift, Louvain community discovery, TreeSHAP feature explanations, and a zero-dependency local serving REST API.
  >
  > Now, let us look specifically at how our heterogeneous transaction graph was constructed."
- **Slide Takeaway:** A 4-stage pipeline separating data ingestion, dual representation, model benchmarking, and operational serving.
- **Transition:** "How did we define the entities and relations in our transaction graph?"

---

### Slide 5: Graph Schema & Grounded Entity Construction
- **Speaking Script:**
  > "A core tenet of our methodology is that graph topology must be grounded in observed data properties, not imagined entities.
  >
  > At the center of our schema is the Transaction Node, representing 590,540 events carrying 470 features and exact seconds timestamps.
  >
  > Connecting to transactions are four distinct entity types:
  > 1. Pseudo-Accounts (199,625 nodes): Real customer identifiers are anonymized in IEEE-CIS. We derived a stable pseudo-account key by combining card number, billing region, and the D1 timedelta offset: card1 + addr1 + (day - D1). The 'charged_to' relation covers 88.9% of transactions.
  > 2. Payment Cards (13,553 nodes): Constructed from card attributes (card1 through card6), connected via 'used_card' with 100% coverage.
  > 3. Network Proxies (2,394 nodes): Infrastructure nodes derived from purchaser email domains and network proxy clusters, with 23.6% coverage via 'via_network'.
  > 4. Device Fingerprints (4,880 nodes): Hardware and operating system combinations (id_30 through id_33) with 12.0% coverage via 'from_device'.
  >
  > Crucially, in Phase 4 we measured and explicitly excluded candidates like raw billing address, email domain, and device strings. These formed massive hubs with millions of connections but yielded zero predictive signal above random chance.
  >
  > Let us now contrast our two approaches to relational feature modeling."
- **Slide Takeaway:** Grounded 5-node, 4-relation schema; non-predictive hub relations were measured and excluded.
- **Transition:** "How do handcrafted relational features compare conceptually against deep graph neural representations?"

---

### Slide 6: Handcrafted Features vs. Learned Graph Representations
- **Speaking Script:**
  > "When applying graph intelligence to tabular streams, practitioners face a fundamental design choice: Do we extract explicit relational features for gradient-boosted trees, or train end-to-end Graph Neural Networks? We implemented and compared both paradigms.
  >
  > On the left is Comparison B: Handcrafted Graph Features.
  > We engineered 40 label-free features capturing backward-looking entity behavior:
  > - Multi-scale velocity counts: Transactions in the preceding 1 day and 7 days per entity.
  > - Recency and monetary volatility: Seconds elapsed since the last transaction, alongside running means and standard deviations of log-transaction amounts.
  > - Relational diversity: The count of distinct payment cards, devices, and networks linked to the pseudo-account to date.
  > All 40 features were locked in PREREGISTRATION_B.md before running any code.
  >
  > On the right are Deep Graph Neural Networks:
  > - Heterogeneous GraphSAGE: Relational neighborhood aggregation with per-node-type linear projections and 64 hidden units.
  > - Graph Attention Networks (GAT): 4-head attention over dynamic neighborhoods.
  > - No-Relations MLP Control: An identical network architecture with all graph edges removed to isolate relational value from neural capacity.
  >
  > Next, let us see how we model temporal entity memory across time."
- **Slide Takeaway:** Contrasting 40 backward-looking handcrafted graph features against deep message-passing GNNs.
- **Transition:** "How do we maintain behavioral context across streaming transactions without looking into the future?"

---

### Slide 7: Modeling Temporal Entity Memory
- **Speaking Script:**
  > "In streaming finance, static graph snapshots miss the evolution of user behavior. A card that lay dormant for six months and suddenly fires 10 high-value transactions in an hour represents a critical behavioral signature.
  >
  > To capture this, we implemented Temporal GraphSAGE with continuous recurrent entity memory.
  > As illustrated in the diagram:
  > - The model executes day-by-day.
  > - Each entity maintains a 64-dimensional latent state updated by a Gated Recurrent Unit (GRU).
  > - When transactions occur on Day $t$, entity states update sequentially from that day's message snapshot and an explicit elapsed-time encoding ($\Delta t$) representing the time since previous activity.
  > - The updated entity state carries forward into the next day without reset.
  >
  > Crucially, in Phase 7A we designed two controls to eliminate confounding factors:
  > 1. Schedule-Matched Static GNN: In standard training, temporal models take ~17 optimizer steps per epoch versus 1 for static GNNs. The schedule-matched control equalizes gradient steps so we don't mistake optimization volume for architectural superiority.
  > 2. Memory-Ablated Temporal GNN: We replace the non-linear recurrent GRU cell with an unweighted cumulative mean, isolating whether recurrent memory dynamics specifically matter.
  >
  > Let us now review the 5 pre-registered hypotheses before I hand over to Presenter B."
- **Slide Takeaway:** Temporal GraphSAGE maintains continuous 64-dim GRU entity states controlled against schedule and memory ablations.
- **Transition:** "Before passing the presentation to my co-presenter, let us formalize what we set out to prove or disprove."

---

### Slide 8: The Core Modeling Strategy & Research Hypotheses
- **Speaking Script:**
  > "To hold our research to strict scientific standards, we pre-registered five formal hypotheses—H1 through H5—with locked falsification criteria before analyzing results:
  > - H1: That graph structure provides predictive lift beyond transaction-level features.
  > - H2: That learned graph neural representations outperform 40 handcrafted graph features.
  > - H3: That temporal entity memory improves future generalization over static models.
  > - H4: That adaptive updating reduces model degradation under concept drift.
  > - H5: That graph clustering uncovers coordinated fraud rings missed by individual transaction scoring.
  >
  > This concludes Part I on our problem formulation, causal hygiene, and system architecture.
  >
  > I will now hand the presentation over to my co-presenter, Presenter B, who will walk you through our experimental validation, quantitative results, surprising baseline comparisons, and scientific conclusions. Presenter B, the floor is yours."
- **Slide Takeaway:** 5 formal pre-registered hypotheses established; clean handoff from Presenter A to Presenter B.
- **Transition:** "Presenter B will now present the experimental evidence and results."

---

# PART II — IMPLEMENTATION & EVIDENCE (PRESENTER B)

### Slide 9: Experimental Protocol & Validation Framework
- **Speaking Script:**
  > "Thank you, Presenter A. I will now present our empirical validation, quantitative findings, and scientific conclusions.
  >
  > When evaluating machine learning on imbalanced fraud datasets, how you evaluate determines whether your conclusions are scientifically valid. We built our validation framework around three core principles:
  >
  > 1. Metric Selection: In an environment with 27.6 benign transactions for every single fraud event, accuracy is meaningless—a trivial baseline predicting all negatives gets 96.5% accuracy. Even ROC-AUC is misleadingly optimistic because a massive denominator of true negatives conceals an unmanageable flood of false alarms. We designate Precision-Recall AUC (PR-AUC) as our primary benchmark because it directly evaluates precision as a function of alert recall.
  >
  > 2. Strict 3-Way Entity Segmentation: Aggregating all transactions into a single metric hides where graph models actually provide value. We evaluate every model across three non-overlapping subsets: Warm transactions with established history, Cold-start transactions where an entity is seen for the first time, and No-Edge transactions. We discovered that conflating cold-start with no-edge transactions artificially inflates reported GNN performance—a practice we strictly barred.
  >
  > 3. Complete Metric Integrity and Sealed Test Period: To ensure absolute reproducibility, we built a consolidated verification table in Phase 12. A single metric engine recomputed all 8,184 recorded metric values with a maximum absolute discrepancy of exactly 0.0. Furthermore, Days 152–182 remain sealed under cryptographic hash guards. Every number presented today is a true validation-period finding.
  >
  > Let us now examine the quantitative comparison between tabular and graph models."
- **Slide Takeaway:** PR-AUC is the true benchmark for imbalanced fraud; strict 3-way segmentation prevents metric inflation; test set remains sealed.
- **Transition:** "Here are our primary empirical findings comparing tabular baselines against graph architectures."

---

### Slide 10: Quantitative Results: Tabular vs. Neural Graph Models
- **Speaking Script:**
  > "Here are our primary empirical findings on the validation period. This analytical chart presents the core quantitative comparison across all models.
  >
  > The primary takeaway is striking: Tuned tabular XGBoost is extraordinarily difficult to beat.
  > - Standard XGBoost on transaction features achieves a PR-AUC of 0.5395 with class weighting, and 0.5948 without class weighting.
  > - Static GraphSAGE scores 0.4637. While it beats its no-relations MLP control (0.4167), it remains far below the tabular model.
  > - Temporal GraphSAGE achieves 0.4927 across three seeds. This is the strongest graph model in our study, yet it still trails baseline XGBoost by nearly 0.047 PR-AUC.
  > - The Graph Attention Network (GAT) never converged within the 150-epoch limit, achieving only 0.3320. We report this openly as a negative result rather than quietly discarding it.
  >
  > Now let us examine Comparison B—adding 40 handcrafted backward-looking graph features to XGBoost:
  > - In the primary single-run comparison, it shows a modest gain of +0.0046 (0.5440 vs 0.5395).
  > - However, when evaluated over three subsampled seeds, the mean difference is -0.0005 against a seed standard deviation of 0.0034.
  > - Under our pre-registered decision rule, H1-direct is NOT ESTABLISHED.
  > - Furthermore, H2 is CONTRADICTED: learned neural graph embeddings do not outperform handcrafted features; both neural models fall far below the handcrafted tabular model.
  >
  > Where, then, does relational and temporal information actually help? Let us examine the ablation studies."
- **Slide Takeaway:** Tuned XGBoost (0.5395/0.5948) beats all neural graph models; H1-direct is not established; H2 is contradicted.
- **Transition:** "Where does relational and temporal structure provide genuine value? Our ablation studies answer this question."

---

### Slide 11: The Power of Temporal Memory & Relation Ablations
- **Speaking Script:**
  > "While neural graph models did not surpass tabular baselines overall, our ablation experiments isolate exactly where relational structure and memory provide genuine lift.
  >
  > First, evaluating Hypothesis H3 (Temporal Memory):
  > When compared against our memory-ablated control—which uses an identical architecture with a cumulative mean—Temporal GraphSAGE wins in 3 of 3 seeds with a mean pooled gain of +0.0137 PR-AUC.
  > When we inspect our 3-way segmentation, the origin of this gain becomes clear: on cold-start transactions, temporal memory provides a substantial +0.027 PR-AUC lift (0.3562 vs 0.3288). When a previously unseen card arrives on a known account, the accumulated GRU state provides critical predictive context. Therefore, H3 is partially supported with a modest, targeted effect.
  >
  > Second, our pre-registered relation-level ablation study on Temporal GraphSAGE:
  > We systematically retrained the temporal model across three seeds with individual relations excised:
  > - Excising the 'charged_to' pseudo-account relation causes a severe drop of -0.0309 PR-AUC across all seeds (falling from 0.4927 to 0.4618), with warm-segment performance plummeting.
  > - Excising the 'from_device' relation causes a modest drop of -0.0073 PR-AUC.
  > - In contrast, excising payment card edges (-0.0038) or network proxy edges (-0.0053) produces differences entirely within our measured run-to-run retraining noise floor of 0.0058.
  >
  > This demonstrates that almost all relational value in the graph model is concentrated in the derived pseudo-account identity key.
  >
  > Now let us turn to another central finding: concept drift and continual adaptation."
- **Slide Takeaway:** Temporal memory lift (+0.027) is concentrated in cold-start transactions; pseudo-account relation drives nearly all structural gain.
- **Transition:** "How do models behave over time when faced with real-world concept drift?"

---

### Slide 12: Concept Drift & Continual Adaptive Updating
- **Speaking Script:**
  > "One of the most consequential contributions of our research is Phase 8: our controlled study of concept drift and walk-forward adaptation over the 30-day validation period.
  >
  > We divided the validation period into five consecutive 6-day windows and tracked three arms initialized from the same day-111 checkpoint:
  > 1. Frozen baseline: Parameters locked at day 111.
  > 2. Continual Adaptive Fine-Tuning: Model warm-start fine-tuned after each window using freshly revealed validation labels.
  > 3. Replay Control: Spends the identical compute budget fine-tuning exclusively on resampled historical training days.
  >
  > Look at the trajectory in Figure 2:
  > The frozen model suffers severe degradation—its PR-AUC plummets from 0.5868 in Window 1 down to 0.3350 in Window 5, a decay of more than 0.25 points in 30 days!
  > Crucially, when we measured covariate shift across all 470 input features, the mean Population Stability Index (PSI) was only 0.01. Raw input distributions barely shifted, yet predictive performance degraded precipitously.
  >
  > Our Continual Adaptive arm maintains substantially higher performance, finishing at 0.3798 with a window-mean PR-AUC of 0.4580 versus 0.4243 for the frozen model.
  > Importantly, our Replay Control scored 0.4179—below the frozen baseline! This proves that the adaptive gain does not stem from extra training steps or compute volume, but strictly from adapting to recent adversarial label distributions.
  > Under our pre-registered decision rules, Hypothesis H4 is fully SUPPORTED.
  >
  > However, we also report a notable negative finding: a pre-registered drift-triggered refresh based on the PSI of the prediction score distribution never fired (PSI stayed below 0.025). Output distribution shift failed to signal label-conditional degradation.
  >
  > Next, let us examine candidate-community discovery and explainability."
- **Slide Takeaway:** Severe performance decay occurs despite low feature drift (PSI ~ 0.01); continual fine-tuning supports H4 (+0.0338 vs. frozen).
- **Transition:** "Beyond transaction classification, can graph structures uncover coordinated fraud rings?"

---

### Slide 13: Candidate-Community Discovery & Explainability
- **Speaking Script:**
  > "In Phase 9 and Phase 11, we investigated coordinated multi-entity fraud discovery and transparent model explainability.
  >
  > First, Candidate Communities:
  > To uncover coordinated groups, we applied Louvain modularity clustering on bipartite card-infrastructure graphs over 7-day windows.
  > Here we demonstrate another example of scientific transparency:
  > In our original Phase 9 benchmark, the synthetic test failed a frozen validity gate (Gate G4) because real transaction ranks shifted. Rather than quietly modifying the gate post-hoc, we adhered to our pre-registration: Hypothesis H5 was marked NOT ESTABLISHED.
  > Subsequently, we pre-registered Phase 9b with an amended interference formula. Phase 9b passed all gates and verified that structure-plus-time recovers camouflaged planted groups with AUPRC 0.14 to 0.27, outperforming baselines (< 0.05).
  > However, on real IEEE-CIS data, we strictly designate clusters as 'candidate coordinated communities'—never 'fraud rings'—because IEEE-CIS contains zero verified fraud ring ground truth.
  >
  > Second, Explainability:
  > Black-box risk scores are unacceptable in financial forensics. We implemented:
  > - Exact TreeSHAP feature contributions for XGBoost, verified to sum to the model margin with an absolute discrepancy under 8.1e-6.
  > - In Comparison B, graph features ranked among the top-10 contributors in 55.3% of evaluated validation transactions.
  > - For Temporal GraphSAGE, counterfactual edge removals demonstrated that excising the pseudo-account edge changed model logits by 3.16 on average.
  > We explicitly communicate that feature attributions describe mathematical model sensitivity, not legal causality.
  >
  > Let us now look at an unexpected finding on class imbalance, and our production software implementation."
- **Slide Takeaway:** Modularity clustering discovers candidate communities; Phase 9b benchmark passed; SHAP & counterfactuals provide transparent explanations.
- **Transition:** "How did class imbalance handling affect our baselines, and how is the production software deployed?"

---

### Slide 14: Class Imbalance Insights & Software Engineering
- **Speaking Script:**
  > "Slide 14 presents two vital aspects of our project: an unexpected machine learning insight on class imbalance, and our production software architecture.
  >
  > On the left is our class imbalance study:
  > Conventional machine learning advice dictates that when faced with 27.6:1 class imbalance, one should set scale_pos_weight = 27.36 to equalize positive and negative loss gradients.
  > However, our pre-registered check revealed a stunning result:
  > Unweighted XGBoost achieves a PR-AUC of 0.5948, outperforming fully weighted XGBoost (0.5395) by +0.0553 points!
  > Why? Full class weighting forces the tree split criteria to aggressively prioritize recall, which severely inflates false-positive alerts at operating thresholds and heavily penalizes precision.
  > In fact, this class weighting decision caused a larger performance swing (+0.0553) than any graph feature (+0.0046) or neural network architecture in our entire study.
  >
  > On the right is our software engineering implementation:
  > This repository is not a collection of disconnected Jupyter notebooks; it is a tested, production-grade system:
  > - In Phase 13, we built an interactive investigation dashboard and REST API using Python's standard library alone—no Flask, FastAPI, or external web frameworks.
  > - We developed an end-to-end demonstration script (demo_end_to_end.py) that generates synthetic transaction streams, computes backward graph features, executes scoring, and detects planted communities on any computer without needing Kaggle credentials.
  > - The repository contains 381 automated tests, including mutation sensitivity tests where 15 of 17 injected bugs were detected.
  >
  > Let us now conclude with our key takeaways and honest limitations."
- **Slide Takeaway:** Unweighted XGBoost beats full class weighting by +0.0553 PR-AUC; production standard-library serving stack and 381 verified tests.
- **Transition:** "To wrap up our presentation, here are our core scientific conclusions and acknowledged limitations."

---

### Slide 15: Conclusions, Limitations & Scientific Takeaways
- **Speaking Script:**
  > "To conclude our presentation, we leave you with three memorable scientific takeaways:
  >
  > 1. The Enduring Power of Tabular Baselines:
  > In financial transaction fraud detection, a properly tuned gradient-boosted tree on raw transaction features remains a formidable benchmark. Many reported GNN gains in the literature vanish once strict causal boundaries and backward-looking leakage protections are enforced.
  >
  > 2. Relational Value is Highly Specific, Not Blanket:
  > Graph and temporal dynamics do contribute real value, but in distinct operational niches:
  > - Continuous recurrent entity memory provides substantial lift (+0.027 PR-AUC) for cold-start transactions where historical account context is otherwise absent.
  > - Continual walk-forward fine-tuning successfully arrests the severe model decay caused by adversarial concept drift.
  >
  > 3. Scientific Hygiene Protects Real-World Systems:
  > Reporting negative results—such as our failed Phase 9 benchmark gate, GAT convergence limits, and the unestablished direct lift of 40 graph features—is vital for sound engineering.
  >
  > Our work has clear, acknowledged boundaries: all numbers reflect the validation period, GNN batch computation remains challenging for sub-100ms streaming inference, and candidate communities require forensic external confirmation.
  >
  > Thank you very much for your time and attention. Both Presenter A and I are now ready to take your questions."
- **Slide Takeaway:** Tabular models remain dominant; relational lift is niche-specific (cold-start & drift); negative results reflect methodological integrity.

---

# LIKELY PROFESSOR DEFENSE QUESTIONS & EVIDENCE-BASED ANSWERS

### Q1: "Why did Graph Neural Networks fail to outperform tabular XGBoost?"
- **Answer:**
  > "Our analysis indicates two primary reasons:
  > 1. Information Redundancy: In the IEEE-CIS dataset, raw transaction features—specifically the engineered count (C), timedelta (D), and Vesta (V) columns—already encode substantial historical entity context. Therefore, relational neighborhood aggregation largely recovers information that gradient boosting already exploits from raw features.
  > 2. Tabular Optimization Dynamics: Gradient-boosted decision trees naturally construct axis-aligned splits on raw numerical features without suffering from the oversmoothing or optimization bottlenecks common to GNN message passing on noisy, heterogeneous financial graphs. The one area where graph memory did provide genuine lift was in cold-start transactions (+0.027 PR-AUC), where historical feature columns are absent."

### Q2: "Did you tune your models on the validation set, and does that cause optimistic bias?"
- **Answer:**
  > "Yes, and we explicitly disclose this in every report and on our slides. All models—including XGBoost early stopping and threshold selection—were tuned on Days 122–151.
  > This is precisely why we instituted our cryptographically sealed held-out protocol for Days 152–182 (`HELDOUT_DECISION_PACKAGE.md`). The held-out period has never been evaluated and awaits formal authorization. By reporting validation results transparently and keeping the test set sealed, we avoid the post-hoc cherry-picking common in academic research."

### Q3: "Why did you call discovered groups 'candidate communities' instead of 'fraud rings'?"
- **Answer:**
  > "Because the IEEE-CIS dataset contains binary transaction-level fraud flags, but zero ground-truth labels for coordinated fraud rings. In Phase 9, our synthetic benchmark proved that Louvain clustering can recover planted coordinated groups. However, asserting that real-world clusters in IEEE-CIS are confirmed criminal rings would be scientifically unfounded. Real-world clusters may simply reflect high-volume benign sharing (e.g., corporate proxy servers or family devices). We maintain strict terminology: they are candidate coordinated communities."

### Q4: "Why can't you serve the Graph Neural Network in your live dashboard?"
- **Answer:**
  > "The neural graph models—HeteroSAGE, GAT, and Temporal GraphSAGE—operate on whole-day graph snapshots with full-batch message passing and recurrent entity state propagation. Serving them in real-time would require an enterprise distributed streaming graph database (such as TigerGraph or Memgraph) to maintain sub-100ms inference latencies. Under the project's minimal-stack constraint, we serve our recorded research XGBoost model, which provides instant millisecond scoring on incoming transactions while preserving exact validation predictions."

### Q5: "Why did class weighting degrade XGBoost performance instead of improving it?"
- **Answer:**
  > "Setting `scale_pos_weight = 27.36` penalizes false negatives 27 times more heavily than false positives during tree splitting. While this drives up raw recall at low thresholds, it shifts the entire score distribution upward and heavily degrades precision at standard operational alerting thresholds (top 1% and 5% alerts). Because PR-AUC integrates precision over all recall operating points, this loss of precision reduces pooled PR-AUC from 0.5948 to 0.5395. Our findings demonstrate that class weighting should never be applied blindly in precision-critical financial environments."
