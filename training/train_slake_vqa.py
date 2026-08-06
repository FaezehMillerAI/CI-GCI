import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config
from utils.slake_loader import SlakeCausalDataset, causal_collate_fn
from utils.vqa_rad_loader import VQARadCausalDataset
from models.cqc_net import CQCNet

def train_vqa(dataset_name="slake", data_dir="data/", config_path="configs/baseline_vqa.yaml", epochs=3, batch_size=16, lr=1e-4, device="cpu"):
    print(f"Starting {dataset_name.upper()} VQA fine-tuning on device: {device}")
    
    # Load configuration
    config = load_config(config_path)
    config["model"]["num_aux_questions"] = 0 # Disable auxiliary tasks for VQA baseline
    config["model"]["num_classes"] = 2       # Binary Yes/No VQA
    
    # Dataset & Loader resolution
    if dataset_name == "slake":
        train_json = os.path.join(data_dir, "slake", "train.json")
        val_json = os.path.join(data_dir, "slake", "validate.json")
        img_dir = os.path.join(data_dir, "slake", "imgs")
        mask_mapping_path = os.path.join(data_dir, "slake", "mask.txt")
        
        train_dataset = SlakeCausalDataset(train_json, img_dir, mask_mapping_path)
        train_dataset.data = [item for item in train_dataset.data if item.get("answer_type") == "CLOSED"]
        
        val_dataset = SlakeCausalDataset(val_json, img_dir, mask_mapping_path)
        val_dataset.data = [item for item in val_dataset.data if item.get("answer_type") == "CLOSED"]
        collate = causal_collate_fn
        
    elif dataset_name == "vqa_rad":
        json_path = os.path.join(data_dir, "VQA-RAD", "VQA_RAD Dataset Public.json")
        img_dir = os.path.join(data_dir, "VQA-RAD", "VQA_RAD Image Folder")
        
        # Load entire closed split and split 80/20 randomly
        full_dataset = VQARadCausalDataset(json_path, img_dir)
        full_dataset.data = [item for item in full_dataset.data if item.get("answer_type") == "CLOSED"]
        
        # Deterministic random split
        g = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(full_dataset), generator=g).tolist()
        split_idx = int(len(indices) * 0.8)
        
        train_dataset = torch.utils.data.Subset(full_dataset, indices[:split_idx])
        val_dataset = torch.utils.data.Subset(full_dataset, indices[split_idx:])
        collate = causal_collate_fn
        
    print(f"Loaded {len(train_dataset)} train and {len(val_dataset)} val closed-ended VQA samples.")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    
    # Initialize Model
    model = CQCNet(config).to(device)
    
    # We load pre-trained weights if available to speed up convergence
    baseline_chk = "outputs/checkpoints/baseline/best_baseline_model.pt"
    if os.path.exists(baseline_chk):
        print(f"Initializing with pre-trained visual/text encoders from {baseline_chk}")
        model.load_state_dict(torch.load(baseline_chk, map_location=device), strict=False)
        
    model.train()
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    ans_map = {"no": 0, "yes": 1}
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        # Training loop
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            images = batch["image"].to(device)
            questions = batch["question"]
            answers = batch["answer"]
            
            # Convert labels to tensor
            labels = torch.tensor([ans_map.get(ans.strip().lower(), 0) for ans in answers], dtype=torch.long, device=device)
            
            optimizer.zero_grad()
            outputs = model(images, questions, device)
            logits = outputs["main_class_logits"]
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            progress_bar.set_postfix(loss=loss.item(), acc=correct/total)
            
        epoch_loss = epoch_loss / total
        train_acc = correct / total
        
        # Validation loop
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                questions = batch["question"]
                answers = batch["answer"]
                
                labels = torch.tensor([ans_map.get(ans.strip().lower(), 0) for ans in answers], dtype=torch.long, device=device)
                outputs = model(images, questions, device)
                logits = outputs["main_class_logits"]
                
                preds = torch.argmax(logits, dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1} - Train Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("models", exist_ok=True)
            # Save specific dataset checkpoint
            checkpoint_path = f"models/{dataset_name}_vqa_model.pth"
            # Fallback copy for backward compatibility
            if dataset_name == "slake":
                torch.save(model.state_dict(), "models/slake_vqa_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model to {checkpoint_path} with Val Acc: {val_acc:.4f}")
            
        model.train()
        scheduler.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="slake", choices=["slake", "vqa_rad"])
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--config_path", type=str, default="configs/baseline_vqa.yaml")
    parser.add_argument("--epochs", type=str, default="3")
    parser.add_argument("--batch_size", type=str, default="16")
    parser.add_argument("--lr", type=str, default="1e-4")
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    
    train_vqa(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        config_path=args.config_path,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        device=args.device
    )
