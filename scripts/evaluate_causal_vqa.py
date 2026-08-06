import os
import sys
import torch
from torch.utils.data import DataLoader
import numpy as np

# Ensure code modules can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config
from utils.slake_loader import SlakeCausalDataset, causal_collate_fn
from models.cqc_net import CQCNet
from models.inpainter import CounterfactualInpainter
from models.causal_decoder import CausalContrastiveDecoder
from evaluation.eval_calibration_grounding import compute_ece
from evaluation.eval_vqa_core import compute_vqa_core_metrics

def run_evaluation(data_dir="data/slake/", config_path="configs/baseline_vqa.yaml", inpainter_path="models/inpainter.pth", device="cpu"):
    print("==================================================")
    print("      CI-GCI PIPELINE CAUSAL EVALUATION LOOP      ")
    print("==================================================")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    # Load configuration
    config = load_config(config_path)
    config["model"]["num_aux_questions"] = 0 # Disable QCG for baseline check
    
    # Initialize VQA Model
    vqa_model = CQCNet(config).to(device)
    slake_chk = "models/slake_vqa_model.pth"
    baseline_chk = "outputs/checkpoints/baseline/best_baseline_model.pt"
    if os.path.exists(slake_chk):
        print(f"Loading SLAKE fine-tuned VQA model from {slake_chk}")
        vqa_model.load_state_dict(torch.load(slake_chk, map_location=device), strict=False)
    elif os.path.exists(baseline_chk):
        print(f"Loading pre-trained VQA baseline from {baseline_chk}")
        vqa_model.load_state_dict(torch.load(baseline_chk, map_location=device), strict=False)
    else:
        print("Warning: pre-trained VQA checkpoint not found, running with random weights.")
        
    vqa_model.eval()
    
    # Initialize Inpainter
    inpainter = CounterfactualInpainter(bilinear=True).to(device)
    if os.path.exists(inpainter_path):
        print(f"Loading trained inpainter from {inpainter_path}")
        inpainter.load_state_dict(torch.load(inpainter_path, map_location=device))
    else:
        print("Warning: trained inpainter checkpoint not found, running with random weights.")
    inpainter.eval()
    
    # Initialize Causal Decoder
    causal_decoder = CausalContrastiveDecoder(gamma=1.5).to(device)
    
    # Load SLAKE dataset (filtered to English closed-ended questions for binary calibration check)
    json_path = os.path.join(data_dir, "test.json")
    img_dir = os.path.join(data_dir, "imgs")
    mask_mapping_path = os.path.join(data_dir, "mask.txt")
    
    # --- Part 1: Closed-Ended VQA Calibration ---
    dataset_closed = SlakeCausalDataset(json_path, img_dir, mask_mapping_path)
    dataset_closed.data = [item for item in dataset_closed.data if item.get("answer_type") == "CLOSED"]
    print(f"Filtered test set to {len(dataset_closed)} English CLOSED-ended questions.")
    
    loader_closed = DataLoader(dataset_closed, batch_size=8, shuffle=False, num_workers=0, collate_fn=causal_collate_fn)
    
    original_preds = []
    calibrated_preds = []
    ground_truths = []
    original_confidences = []
    calibrated_confidences = []
    
    ans_map = {"no": 0, "yes": 1}
    inv_ans_map = {0: "no", 1: "yes"}
    
    with torch.no_grad():
        for batch in loader_closed:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            questions = batch["question"]
            answers = batch["answer"]
            
            batch_gts = [ans_map.get(ans.strip().lower(), 0) for ans in answers]
            ground_truths.extend(batch_gts)
            
            # Forward VQA pass on original and counterfactual images
            original_outputs = vqa_model(images, questions, device)
            original_logits = original_outputs["main_class_logits"]
            
            counterfactual_images = inpainter(images, masks)
            counterfactual_outputs = vqa_model(counterfactual_images, questions, device)
            counterfactual_logits = counterfactual_outputs["main_class_logits"]
            
            # Calibrate via Causal Contrastive Decoder
            causal_out = causal_decoder(original_logits, counterfactual_logits)
            calibrated_probs = causal_out["calibrated_probs"]
            
            orig_probs = torch.softmax(original_logits, dim=-1)
            orig_pred_classes = torch.argmax(orig_probs, dim=-1).cpu().numpy()
            cal_pred_classes = torch.argmax(calibrated_probs, dim=-1).cpu().numpy()
            
            orig_conf = orig_probs[:, 1].cpu().numpy()
            cal_conf = calibrated_probs[:, 1].cpu().numpy()
            
            original_preds.extend([inv_ans_map[p] for p in orig_pred_classes])
            calibrated_preds.extend([inv_ans_map[p] for p in cal_pred_classes])
            original_confidences.extend(orig_conf)
            calibrated_confidences.extend(cal_conf)
            
    # Calculate Closed Metrics
    original_gts_str = [inv_ans_map[gt] for gt in ground_truths]
    orig_vqa = compute_vqa_core_metrics(original_preds, original_gts_str)
    cal_vqa = compute_vqa_core_metrics(calibrated_preds, original_gts_str)
    
    orig_ece, orig_mce = compute_ece(np.array(original_confidences), np.array(ground_truths))
    cal_ece, cal_mce = compute_ece(np.array(calibrated_confidences), np.array(ground_truths))
    
    # --- Part 2: Open-Ended VQA (Generative) Calibration ---
    dataset_open = SlakeCausalDataset(json_path, img_dir, mask_mapping_path)
    dataset_open.data = [item for item in dataset_open.data if item.get("answer_type") == "OPEN"][:100] # Limit to 100 samples for speed
    print(f"Filtered test set to {len(dataset_open)} English OPEN-ended questions.")
    
    loader_open = DataLoader(dataset_open, batch_size=8, shuffle=False, num_workers=0, collate_fn=causal_collate_fn)
    
    gen_diffs = []
    with torch.no_grad():
        for batch in loader_open:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            questions = batch["question"]
            
            # Forward VQA passes
            original_outputs = vqa_model(images, questions, device)
            original_gen_logits = original_outputs["main_gen_logits"] # (B, vocab_size)
            
            counterfactual_images = inpainter(images, masks)
            counterfactual_outputs = vqa_model(counterfactual_images, questions, device)
            counterfactual_gen_logits = counterfactual_outputs["main_gen_logits"]
            
            # Calibrate vocabulary logits
            calibrated_gen_logits = causal_decoder.calibrate_generative_logits(original_gen_logits, counterfactual_gen_logits)
            
            # Measure difference
            diff = torch.mean(torch.abs(original_gen_logits - calibrated_gen_logits)).item()
            gen_diffs.append(diff)
            
    avg_gen_diff = np.mean(gen_diffs) if gen_diffs else 0.0
    
    print("\n==================================================")
    print("                EVALUATION SUMMARY                ")
    print("==================================================")
    print("Closed-Ended Questions:")
    print(f"  Original VQA Accuracy:              {orig_vqa['accuracy']:.4f}")
    print(f"  Calibrated Causal VQA Accuracy:     {cal_vqa['accuracy']:.4f}")
    print(f"  Original ECE (Calibration Error):   {orig_ece:.4f}")
    print(f"  Calibrated ECE (Calibration Error): {cal_ece:.4f}")
    print("Open-Ended Questions:")
    print(f"  Causal Generative Logit Shift:     {avg_gen_diff:.6f}")
    print("==================================================")

if __name__ == "__main__":
    run_evaluation()
