import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config
from utils.slake_loader import SlakeCausalDataset, causal_collate_fn
from models.cqc_net import CQCNet

def train_slake_vqa(data_dir="data/slake/", config_path="configs/baseline_vqa.yaml", epochs=3, batch_size=16, lr=1e-4, device="cpu"):
    print(f"Starting SLAKE VQA fine-tuning on device: {device}")
    
    # Load configuration
    config = load_config(config_path)
    config["model"]["num_aux_questions"] = 0 # Disable auxiliary tasks for VQA baseline
    config["model"]["num_classes"] = 2       # Binary Yes/No VQA
    
    # Dataset & Loader
    train_json = os.path.join(data_dir, "train.json")
    val_json = os.path.join(data_dir, "validate.json")
    img_dir = os.path.join(data_dir, "imgs")
    mask_mapping_path = os.path.join(data_dir, "mask.txt")
    
    train_dataset = SlakeCausalDataset(train_json, img_dir, mask_mapping_path)
    train_dataset.data = [item for item in train_dataset.data if item.get("answer_type") == "CLOSED"]
    
    val_dataset = SlakeCausalDataset(val_json, img_dir, mask_mapping_path)
    val_dataset.data = [item for item in val_dataset.data if item.get("answer_type") == "CLOSED"]
    
    print(f"Loaded {len(train_dataset)} train and {len(val_dataset)} val closed-ended VQA samples.")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=causal_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=causal_collate_fn)
    
    # Initialize Model
    model = CQCNet(config).to(device)
    
    # We can load pre-trained weights if available, to speed up convergence
    baseline_chk = "outputs/checkpoints/baseline/best_baseline_model.pt"
    if os.path.exists(baseline_chk):
        print(f"Initializing with pre-trained visual/text encoders from {baseline_chk}")
        model.load_state_dict(torch.load(baseline_chk, map_location=device), strict=False)
        
    model.train()
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    ans_map = {"no": 0, "yes": 1}
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in loop:
            images = batch["image"].to(device)
            questions = batch["question"]
            answers = batch["answer"]
            
            # Map answers to binary labels
            labels = torch.tensor([ans_map.get(ans.strip().lower(), 0) for ans in answers], dtype=torch.long, device=device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images, questions, device)
            logits = outputs["main_class_logits"]
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            loop.set_postfix(loss=loss.item(), acc=(correct/total))
            
        # Validation
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
                
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        print(f"Epoch {epoch+1} - Train Loss: {epoch_loss/len(train_loader):.4f} | Train Acc: {correct/total:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/slake_vqa_model.pth")
            print(f"Saved best model with Val Acc: {best_val_acc:.4f}")

if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    train_slake_vqa(device=device, epochs=3, batch_size=16)
