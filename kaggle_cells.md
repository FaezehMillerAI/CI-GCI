# Kaggle Notebook Cells for CI-GCI Pipeline

You can copy and paste the following blocks directly into your Kaggle Notebook cells to run the entire training and causal evaluation loop.

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

### **Cell 2: Dataset Configuration (SLAKE)**
This cell instantly links the pre-mounted Kaggle SLAKE dataset to the expected folder path using a symbolic link (takes 0 seconds and uses 0MB of disk):

```python
import os

# Create data directory structure
os.makedirs("data", exist_ok=True)

# Path to the mounted Kaggle dataset
kaggle_dataset_path = "/kaggle/input/datasets/rounakmandal/slake-vqa/Slake1.0"
target_link = "data/slake"

if os.path.exists(target_link):
    # Remove old link or folder if present
    if os.path.islink(target_link):
        os.unlink(target_link)
    else:
        import shutil
        shutil.rmtree(target_link)

if os.path.exists(kaggle_dataset_path):
    print(f"Dataset found at {kaggle_dataset_path}. Linking to data/slake...")
    os.symlink(kaggle_dataset_path, target_link)
    print("Symbolic link created successfully!")
else:
    print(f"Error: Could not find dataset at {kaggle_dataset_path}.")
    print("Please double check your Kaggle dataset mount name.")
```

---

### **Cell 3: Train the Counterfactual Inpainter**
Trains the UNet generative network on SLAKE images. (This will run on the Kaggle GPU/CUDA device).

```python
# Train the Counterfactual Inpainter
!PYTHONPATH=. python3 training/train_inpainter.py --epochs 5 --batch_size 16 --device cuda
```

---

### **Cell 4: Fine-tune VQA Model on SLAKE**
Finetunes the visual and text encoders on SLAKE closed-ended VQA tasks to align representation spaces.

```python
# Fine-tune VQA model
!PYTHONPATH=. python3 training/train_slake_vqa.py --epochs 3 --batch_size 16 --device cuda
```

---

### **Cell 5: Run Causal Evaluation & Comparative Benchmarks**
Calculates Accuracy and Expected Calibration Error (ECE), comparing the uncalibrated baseline against our CI-GCI pipeline.

```python
# Run benchmarks
!PYTHONPATH=. python3 scripts/benchmark_comparison.py
```

---

### **Cell 6: Generate & Display Visual Proof Sheets**
Generates the reliability diagrams and prints side-by-side scan comparisons.

```python
# 1. Run plotting script
!PYTHONPATH=. python3 scripts/generate_plots_and_proofs.py

# 2. Display the reliability diagram inside the notebook
from IPython.display import Image, display

print("--- Reliability Diagrams (Calibration Improvement) ---")
display(Image(filename="outputs/reliability_diagram.png"))

print("\n--- Visual Proof Sheet: Patient 10 ---")
display(Image(filename="outputs/proofs/proof_sample_10.png"))

print("\n--- Visual Proof Sheet: Patient 12 ---")
display(Image(filename="outputs/proofs/proof_sample_12.png"))
```
