# Kaggle Notebook Cells: Mature Multi-Dataset CI-GCI Pipeline

Copy and paste the following cells into your Kaggle Notebook. This version dynamically supports **SLAKE**, **VQA-RAD**, **MS-CXR**, and **HEAL-MedVQA**.

---

### **Cell 1: Clone Repository & Setup Environment**
Run this block to clone the codebase and install requirements.

```python
# 1. Clone the repository
!git clone https://github.com/FaezehMillerAI/CI-GCI.git
%cd CI-GCI

# 2. Install requirements
!pip install -r requirements.txt
```

---

### **Cell 2: Dataset Configuration & Symbolic Linking**
Creates links to mounted Kaggle dataset files so the code registers their paths immediately:

```python
import os

# Create data directories
os.makedirs("data", exist_ok=True)

# 1. Link SLAKE dataset
slake_input_path = "/kaggle/input/datasets/rounakmandal/slake-vqa/Slake1.0"
slake_link = "data/slake"
if os.path.exists(slake_input_path):
    if not os.path.exists(slake_link):
        os.symlink(slake_input_path, slake_link)
        print("SLAKE dataset linked successfully!")
else:
    print("SLAKE dataset path not found. Checking fallback...")

# 2. Link MS-CXR dataset (adjust based on your Kaggle upload name)
ms_cxr_input_path = "/kaggle/input/ms-cxr"
ms_cxr_link = "data/ms-cxr"
if os.path.exists(ms_cxr_input_path) and not os.path.exists(ms_cxr_link):
    os.symlink(ms_cxr_input_path, ms_cxr_link)
    print("MS-CXR linked successfully!")

print("Environment dataset directories ready.")
```

---

### **Cell 3: Run Multi-Dataset Smoke Test**
Run this cell to immediately verify that VQA-RAD, MS-CXR, HEAL-MedVQA, and SLAKE loaders and forward passes are working correctly.

```python
# Run the end-to-end smoke test on all datasets
!PYTHONPATH=. python3 scripts/smoke_test_all.py --device cuda
```

---

### **Cell 4: Train the Counterfactual Inpainter**
Trains the UNet generative network on SLAKE images (runs on GPU).

```python
# Train the Counterfactual Inpainter on GPU (saves to models/inpainter.pth)
!PYTHONPATH=. python3 training/train_inpainter.py --epochs 5 --batch_size 16 --device cuda
```

---

### **Cell 5: VQA Model Fine-Tuning**
Select which dataset you wish to train on by setting the `--dataset` argument:

```python
# Option A: Fine-tune on SLAKE
!PYTHONPATH=. python3 training/train_slake_vqa.py --dataset slake --epochs 3 --batch_size 16 --device cuda

# Option B: Fine-tune on VQA-RAD (Uncomment if training VQA-RAD)
# !PYTHONPATH=. python3 training/train_slake_vqa.py --dataset vqa_rad --epochs 3 --batch_size 16 --device cuda
```

---

### **Cell 6: Run Comparative Benchmarks (SOTA Study)**
Evaluates and compares the uncalibrated baseline against our CI-GCI pipeline across all datasets.

```python
print("--- BENCHMARK RESULTS: SLAKE ---")
!PYTHONPATH=. python3 scripts/benchmark_comparison.py --dataset slake --device cuda

print("\n--- BENCHMARK RESULTS: VQA-RAD ---")
!PYTHONPATH=. python3 scripts/benchmark_comparison.py --dataset vqa_rad --device cuda

print("\n--- BENCHMARK RESULTS: MS-CXR ---")
!PYTHONPATH=. python3 scripts/benchmark_comparison.py --dataset ms_cxr --device cuda

print("\n--- BENCHMARK RESULTS: HEAL-MedVQA ---")
!PYTHONPATH=. python3 scripts/benchmark_comparison.py --dataset heal --device cuda
```

---

### **Cell 7: Generate & Display Visual Proof Sheets**
Generates the reliability diagrams and side-by-side scan comparisons.

```python
# 1. Run plotting script for SLAKE
!PYTHONPATH=. python3 scripts/generate_plots_and_proofs.py --dataset slake --device cuda

# 2. Run plotting script for VQA-RAD
!PYTHONPATH=. python3 scripts/generate_plots_and_proofs.py --dataset vqa_rad --device cuda

# 3. Display the generated plots directly inside the Kaggle notebook
from IPython.display import Image, display

print("--- SLAKE Reliability Diagrams ---")
display(Image(filename="outputs/reliability_diagram_slake.png"))

print("\n--- VQA-RAD Reliability Diagrams ---")
display(Image(filename="outputs/reliability_diagram_vqa_rad.png"))

print("\n--- Visual Proof Sheet: SLAKE Patient 0 ---")
display(Image(filename="outputs/proofs/proof_slake_sample_0.png"))
```
