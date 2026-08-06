import os
import sys
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Ensure code modules can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config
from utils.slake_loader import SlakeCausalDataset, causal_collate_fn
from models.cqc_net import CQCNet
from models.inpainter import CounterfactualInpainter
from models.causal_decoder import CausalContrastiveDecoder

def generate_reliability_diagram(original_confidences, calibrated_confidences, ground_truths, save_path="outputs/reliability_diagram.png", num_bins=10):
    """
    Plots Reliability Diagrams (confidence vs. accuracy) for original and calibrated models.
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    
    orig_accs = []
    orig_confs = []
    cal_accs = []
    cal_confs = []
    
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Original Bin Stats
        in_bin_orig = (original_confidences >= bin_lower) & (original_confidences < bin_upper)
        if i == num_bins - 1:
            in_bin_orig = in_bin_orig | (original_confidences == bin_upper)
        if np.sum(in_bin_orig) > 0:
            orig_accs.append(np.mean(ground_truths[in_bin_orig]))
            orig_confs.append(np.mean(original_confidences[in_bin_orig]))
        else:
            orig_accs.append(0.0)
            orig_confs.append((bin_lower + bin_upper) / 2.0)
            
        # Calibrated Bin Stats
        in_bin_cal = (calibrated_confidences >= bin_lower) & (calibrated_confidences < bin_upper)
        if i == num_bins - 1:
            in_bin_cal = in_bin_cal | (calibrated_confidences == bin_upper)
        if np.sum(in_bin_cal) > 0:
            cal_accs.append(np.mean(ground_truths[in_bin_cal]))
            cal_confs.append(np.mean(calibrated_confidences[in_bin_cal]))
        else:
            cal_accs.append(0.0)
            cal_confs.append((bin_lower + bin_upper) / 2.0)
            
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Original
    axes[0].bar(bin_boundaries[:-1], orig_accs, width=1.0/num_bins, align='edge', color='red', alpha=0.6, edgecolor='red', label='Outputs')
    axes[0].plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])
    axes[0].set_xlabel("Confidence")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Original VQA Calibration")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    # Right: Calibrated
    axes[1].bar(bin_boundaries[:-1], cal_accs, width=1.0/num_bins, align='edge', color='green', alpha=0.6, edgecolor='green', label='Outputs')
    axes[1].plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1])
    axes[1].set_xlabel("Confidence")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Calibrated Causal VQA Calibration")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Reliability Diagram saved successfully to {save_path}")

def generate_proof_sheets(dataset, inpainter, device, save_dir="outputs/proofs/"):
    """
    Saves visual side-by-side columns: Original Image, ROI Mask, and Inpainted Image.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Pick a few distinct samples with non-empty masks
    sampled_indices = []
    for idx in range(len(dataset)):
        mask = dataset[idx]["mask"]
        if mask.sum() > 200: # Has a significant pathology / organ region
            sampled_indices.append(idx)
        if len(sampled_indices) >= 3:
            break
            
    print(f"Generating visual proofs for test indices: {sampled_indices}")
    
    for idx in sampled_indices:
        item = dataset[idx]
        image = item["image"].unsqueeze(0).to(device)
        mask = item["mask"].unsqueeze(0).to(device)
        
        with torch.no_grad():
            cf_image = inpainter(image, mask)
            
        # Denormalize image for display
        img_np = (image[0].cpu().numpy().transpose(1, 2, 0) * 0.229 + 0.485).clip(0, 1)
        cf_np = (cf_image[0].cpu().numpy().transpose(1, 2, 0) * 0.229 + 0.485).clip(0, 1)
        mask_np = mask[0, 0].cpu().numpy()
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(img_np)
        axes[0].set_title("Original Scan")
        axes[0].axis('off')
        
        axes[1].imshow(mask_np, cmap='gray')
        axes[1].set_title(f"Target Mask M\n({', '.join(item['matched_classes'])})")
        axes[1].axis('off')
        
        axes[2].imshow(cf_np)
        axes[2].set_title("Inpainted Healthy Scan")
        axes[2].axis('off')
        
        save_path = os.path.join(save_dir, f"proof_sample_{idx}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Proof Sheet saved successfully to {save_path}")

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    data_dir = "data/slake/"
    json_path = os.path.join(data_dir, "test.json")
    img_dir = os.path.join(data_dir, "imgs")
    mask_mapping_path = os.path.join(data_dir, "mask.txt")
    
    dataset = SlakeCausalDataset(json_path, img_dir, mask_mapping_path)
    dataset.data = [item for item in dataset.data if item.get("answer_type") == "CLOSED"]
    
    # Load VQA and Inpainter
    config = load_config("configs/baseline_vqa.yaml")
    config["model"]["num_aux_questions"] = 0
    vqa_model = CQCNet(config).to(device)
    
    slake_chk = "models/slake_vqa_model.pth"
    if os.path.exists(slake_chk):
        vqa_model.load_state_dict(torch.load(slake_chk, map_location=device), strict=False)
        
    inpainter = CounterfactualInpainter(bilinear=True).to(device)
    if os.path.exists("models/inpainter.pth"):
        inpainter.load_state_dict(torch.load("models/inpainter.pth", map_location=device))
        
    causal_decoder = CausalContrastiveDecoder(gamma=1.5).to(device)
    
    vqa_model.eval()
    inpainter.eval()
    
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=causal_collate_fn)
    
    original_confidences = []
    calibrated_confidences = []
    ground_truths = []
    ans_map = {"no": 0, "yes": 1}
    
    # Generate proof sheets first
    generate_proof_sheets(dataset, inpainter, device)
    
    # Collect predictions to generate reliability diagram
    print("Collecting calibration logits...")
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            questions = batch["question"]
            answers = batch["answer"]
            
            batch_gts = [ans_map.get(ans.strip().lower(), 0) for ans in answers]
            ground_truths.extend(batch_gts)
            
            original_outputs = vqa_model(images, questions, device)
            original_logits = original_outputs["main_class_logits"]
            
            cf_images = inpainter(images, masks)
            cf_outputs = vqa_model(cf_images, questions, device)
            cf_logits = cf_outputs["main_class_logits"]
            
            causal_out = causal_decoder(original_logits, cf_logits)
            calibrated_probs = causal_out["calibrated_probs"]
            
            orig_probs = torch.softmax(original_logits, dim=-1)
            original_confidences.extend(orig_probs[:, 1].cpu().numpy())
            calibrated_confidences.extend(calibrated_probs[:, 1].cpu().numpy())
            
    # Generate reliability diagram
    generate_reliability_diagram(
        np.array(original_confidences),
        np.array(calibrated_confidences),
        np.array(ground_truths)
    )

if __name__ == "__main__":
    main()
