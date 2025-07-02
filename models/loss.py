from torch.nn.functional import mse_loss, cross_entropy, softmin, softmax
import torch
import torch.nn as nn
class dist_softmax(nn.Module):
    def __init__(self):
        super(dist_softmax, self).__init__()

    def forward(self, pred, target):
        """
        Forward pass for the distance softmax loss.
        
        Args:
            pred (torch.Tensor): Predicted values.
            target (torch.Tensor): Target values.
            
        Returns:
            torch.Tensor: Computed loss value.
        """
        B, M, _ = pred.size()  # B: batch size, M: sequence length, _: number of branches (5 in this case)
        top_probs = pred[..., 0]  # shape: [B, S, 5]
        # Step 2: Find index of max probability across 5 branches
        max_idx = top_probs.argmax(dim=1) 
        B_idx = torch.arange(B)

        selected = pred[B_idx, max_idx, 1:]  # shape: [B, S, 1025]

        probs = softmax(pred[:,:,1:], dim=-1)
        topk_vals = torch.argmax(probs, dim=-1)  # Get the indices of the topk values
        distance = torch.abs(topk_vals.squeeze(-1) - target.unsqueeze(1))  # [B, Nnodes, output_size]
        weights = softmin(distance / 1, dim=-1)



        entropyloss0 = cross_entropy(pred[:,:,0], weights, reduction='mean')
        entropyloss1 = cross_entropy(selected, target, reduction='mean')


        return entropyloss0 + entropyloss1 # Combine the two losses with a weight factor

