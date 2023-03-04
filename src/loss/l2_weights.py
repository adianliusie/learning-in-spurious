import torch
import torch.nn.functional as F

from types import SimpleNamespace
from typing import Tuple
from copy import deepcopy

from .base import BaseLoss

class L2WeightRegLoss(BaseLoss):
    """ Regularlisation Loss to avoid catstrophic forgetting- adds L2 distance of weights"""
    def __init__(self, model, alpha=1):
        super().__init__()
        # save access to main model
        self.model = model
        
        # copy initial model
        self.init_model = deepcopy(model)
        
        # set weighting hyperparameter
        self.alpha = alpha

    def forward(self, batch: SimpleNamespace) -> Tuple[float, dict]:
        output = self.model(
            input_ids = batch.input_ids, 
            attention_mask = batch.attention_mask, 
            mask_positions = batch.mask_positions
        )

        # Cross entropy loss
        ce_loss = F.cross_entropy(output.logits, batch.labels)

        # Calculate model parameters L2 loss
        L2_reg = 0
        for param1, param2 in zip(self.model.parameters(), self.init_model.parameters()):
            L2_reg += torch.sum((param1 - param2)**2)

        # Get overall loss
        loss = ce_loss + self.alpha * L2_reg

        # Masking out all non-labels
        hits = torch.argmax(output.logits, dim=-1) == batch.labels
        acc = hits.sum()/len(batch.labels)

        #record training metrics
        self.record_metrics({
            'loss': loss.item(),
            'ce_loss':ce_loss.item(),
            'L2_reg':L2_reg.item(),
            'acc': acc.item(),
            'select': acc.item(),
        })

        return SimpleNamespace(
                    loss=loss, 
                    ce_loss=ce_loss,
                    L2_reg=L2_reg,
                    logits=output.logits, 
                    h=output.h
        )

    def to(self, device):
        self.init_model.to(device)
