# Implementation Plan: CI-GCI Advanced Features & Evaluation

We outline the plan to expand the **CI-GCI** framework from a binary closed-ended prototype into a complete, publishable research project. This includes open-ended VQA generative decoding, plotting tools, and comparative benchmarking.

---

## Proposed Changes

### Component 1: Open-Ended Generative VQA Causal Calibration

#### [MODIFY] [causal_decoder.py](file:///Users/fs525/Desktop/CQC2/models/causal_decoder.py)
*   **Purpose:** Extend the Causal Contrastive Decoder to support open-ended text answers.
*   **Details:** Instead of comparing class logits, we will compare the text generation cross-entropy loss (perplexity) of the predicted answer string when conditioned on the original scan vs. the counterfactual healthy scan:
    \[ \Delta \mathcal{L} = \mathcal{L}(A | I_{cf}, Q) - \mathcal{L}(A | I, Q) \]
    If removing the pathology doesn't increase generation loss (meaning the VLM would generate the same diagnosis even without the pathology), it flags a sequence-level hallucination.

---

### Component 2: Visualization and Proof-Sheet Generation

#### [NEW] [generate_plots_and_proofs.py](file:///Users/fs525/Desktop/CQC2/scripts/generate_plots_and_proofs.py)
*   **Purpose:** Automates creation of publication-ready figures.
*   **Details:**
    1.  **Reliability Diagrams:** Plots bin confidence vs. accuracy curves for both original and calibrated causal VQA to visually demonstrate ECE calibration improvements.
    2.  **Inference-Time Proof Sheets:** Saves side-by-side PNG comparisons of the Original Scan, the targeted ROI Mask ($M$), the Inpainted Scan, and the Causal Logit Difference ($\Delta Z$).

---

### Component 3: Comparative Benchmarks

#### [NEW] [benchmark_comparison.py](file:///Users/fs525/Desktop/CQC2/scripts/benchmark_comparison.py)
*   **Purpose:** Evaluates and compares CI-GCI against standard attention-guided calibration baselines.
*   **Baselines:**
    1.  *Standard VQA*: Uncalibrated model predictions.
    2.  *Attention-Saliency Filtering*: Calibrating predictions based on raw cross-attention weights rather than generative image modification.
    3.  *CI-GCI (Ours)*: Generative counterfactual inpainting loop.

---

## Verification Plan

### Automated Tests
*   `python scripts/generate_plots_and_proofs.py` to verify that all reliability diagrams and proof-sheet images are successfully saved.
*   `python scripts/benchmark_comparison.py` to verify that comparison results are formatted into a markdown table.
