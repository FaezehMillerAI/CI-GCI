import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalContrastiveDecoder(nn.Module):
    """
    Causal Contrastive Decoder (CCD) filters out linguistic shortcuts by
    adjusting output logits using the Individual Causal Effect (ICE).
    """
    def __init__(self, gamma=1.5):
        super().__init__()
        self.gamma = gamma

    def forward(self, original_logits, counterfactual_logits):
        """
        Calibrates logits based on visual causal influence.
        Args:
            original_logits (torch.Tensor): Logits from original image (B, num_classes)
            counterfactual_logits (torch.Tensor): Logits from counterfactual healthy image (B, num_classes)
        Returns:
            dict: {
                "calibrated_probs": calibrated probabilities,
                "ice": Individual Causal Effect,
                "hallucination_score": hallucination probability
            }
        """
        # 1. Calculate Individual Causal Effect (ICE)
        # For positive visual features, removing the pathology should decrease logit value
        ice = original_logits - counterfactual_logits
        
        # 2. Estimate Hallucination Risk
        # If ICE is small or negative (meaning removing pathology did NOT decrease confidence),
        # then the prediction is driven by language shortcut / bias.
        # We calculate the deviation from expected causal drop:
        hallucination_score = torch.sigmoid(-self.gamma * ice)
        
        # 3. Calibrate Probabilities
        # Original probabilities
        orig_probs = F.softmax(original_logits, dim=-1)
        
        # Scale original probabilities down where hallucination risk is high
        calibrated_probs = orig_probs * (1.0 - hallucination_score)
        
        # Re-normalize to sum to 1
        calibrated_probs = F.normalize(calibrated_probs, p=1, dim=-1)
        
        return {
            "calibrated_probs": calibrated_probs,
            "ice": ice,
            "hallucination_score": hallucination_score
        }

    def calibrate_generative_logits(self, original_gen_logits, counterfactual_gen_logits):
        """
        Calibrates vocabulary token logits for open-ended text VQA generation.
        Args:
            original_gen_logits (torch.Tensor): Gen logits from original image (B, vocab_size)
            counterfactual_gen_logits (torch.Tensor): Gen logits from counterfactual scan (B, vocab_size)
        Returns:
            torch.Tensor: Calibrated generative logits
        """
        ice = original_gen_logits - counterfactual_gen_logits
        hallucination_penalty = torch.sigmoid(-self.gamma * ice)
        calibrated_gen_logits = original_gen_logits - self.gamma * hallucination_penalty
        return calibrated_gen_logits

