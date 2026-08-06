# CQC-Net: Counterfactual Question Curriculum Network for Hallucination-Aware Medical VQA

CQC-Net is a novel framework designed to mitigate hallucinations in Medical Visual Question Answering (Med-VQA). It generates a structured 3-level question curriculum (existence/localization, attribute/relation, clinical inference), computes answers, evaluates grounding against visual evidence, and assesses hallucination risk via hierarchical and directional inconsistency.

---

## Repository Structure

```text
configs/                     # YAML configuration files
  default_config.yaml        # Main settings
  baseline_vqa.yaml          # Stage 1 training configuration
  joint_training.yaml        # Stage 4 joint training configuration
  ablation_configs/          # Configurations for ablation experiments
data/                        # Raw, processed, and curriculum dataset directories
models/                      # Modular architecture components
  visual_encoder.py          # Dual-scale ResNet/DenseNet/ViT/Swin encoders
  text_encoder.py            # Modular PubMedBERT / BioClinicalBERT encoders
  qcg.py                     # Question Curriculum Generator (QCG)
  answer_generator.py        # VQA answer generation module
  verifier.py                # Grounding & Evidence Verifier
  consistency_head.py        # GRU-based Directional Inconsistency Detector
  refiner.py                 # Abstention and refinement routing module
  cqc_net.py                 # Top-level unified wrappers
training/                    # Trainer classes and multi-task loss functions
  dataset.py                 # VQA-RAD/SLAKE replication data loader
  loss_functions.py          # QA, Grounding, Consistency, Hallucination, and Brier calibration loss
  trainer_baseline.py        # Baseline model trainer
  trainer_qcg.py             # Curriculum generator trainer
  trainer_joint.py           # Multi-task joint trainer
evaluation/                  # Diagnostic metrics suite
  eval_vqa_core.py           # Accuracy, EM, F1
  eval_nlg.py                # BLEU, ROUGE, METEOR, CIDEr, BERTScore
  eval_hallucination.py      # Hallucination rate, AUROC, AUPRC, FPR@95
  eval_calibration_grounding.py # ECE, MCE, Brier, selective prediction (coverage & risk)
  eval_qcg.py                # QCG relevance and diversity
  result_table_generator.py  # Generates paper-ready CSV and Markdown summary tables
scripts/                     # Command-line executables
  build_synthetic_data.py    # Zero-dependency synthetic pipeline generator
  build_curriculum_data.py   # Hybrid question-chain augmentor
  train.py                   # Dispatcher for training stages
  evaluate.py                # Metric evaluation dispatcher
  infer.py                   # End-to-end inference and grounding bbox visualizer
utils/                       # Helper functions (seed, config loaders)
requirements.txt             # Project library requirements
```

---

## Getting Started

### 1. Installation

Install all required libraries:
```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Dataset & Curriculum

To instantly dry-run the codebase without external dataset downloads:
```bash
# Generate mock images and metadata
python3 scripts/build_synthetic_data.py --num_samples 50 --output_dir data/processed/synthetic

# Build the 3-level question curriculum
python3 scripts/build_curriculum_data.py \
  --input_data data/processed/synthetic/dataset.json \
  --output_data data/curriculum/synthetic_curriculum.json \
  --mode hybrid
```

---

## Training Pipeline

CQC-Net is trained in sequential research stages:

### Stage 1: Baseline Med-VQA Training
Trains the visual/text encoders and base Answer Generator:
```bash
python3 scripts/train.py --stage baseline --config configs/baseline_vqa.yaml
```

### Stage 2: QCG Training
Freezes encoders and trains the curriculum generator to output level-consistent auxiliary question representations:
```bash
python3 scripts/train.py --stage qcg --config configs/default_config.yaml
```

### Stage 3: Joint Training
Trains the Answerer, Verifier, and GRU-based Inconsistency Detector:
```bash
python3 scripts/train.py --stage joint --config configs/joint_training.yaml
```

### Run All Stages Sequentially
```bash
python3 scripts/train.py --stage all --config configs/default_config.yaml
```

---

## Evaluation & Results

Run the full metrics evaluation suite on the test set. This automatically aggregates predictions and outputs publication-ready tables under `outputs/tables/`:
```bash
python3 scripts/evaluate.py --config configs/default_config.yaml
```

---

## Inference & Bounding Box Grounding Visualization

Run inference on any medical image to get the VQA answer, curriculum questions, grounding scores, hallucination risk, and the decision flag:
```bash
python3 scripts/infer.py \
  --image data/processed/synthetic/images/sample_0.png \
  --question "Is there evidence of pleural effusion?" \
  --output_viz outputs/inference_grounding.png
```
This saves a visualization image showing the predicted region of interest bounding box.
