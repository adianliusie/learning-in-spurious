import torch
import torch.nn.functional as F

from types import SimpleNamespace
from typing import Tuple
from copy import deepcopy

from .base import BaseLoss

class KlLogitRegLoss(BaseLoss):
    """ Regularlisation Loss to avoid catstrophic forgetting- add KL logits loss (to start)"""
    def __init__(self, model, alpha=1, vocab:bool=False):
        super().__init__()
        # save access to main model
        self.model = model
        
        # copy initial model
        self.init_model = deepcopy(model)
        
        # set seeting and weighting hyperparameter
        self.use_vocab_logits = vocab
        self.alpha = alpha

    def forward(self, batch: SimpleNamespace) -> Tuple[float, dict]:
        output = self.model(
            input_ids = batch.input_ids, 
            attention_mask = batch.attention_mask, 
            mask_positions = batch.mask_positions
        )

        init_output = self.init_model(
            input_ids = batch.input_ids, 
            attention_mask = batch.attention_mask, 
            mask_positions = batch.mask_positions
        )

        # Cross entropy loss
        ce_loss = F.cross_entropy(output.logits, batch.labels)

        # Calculate initial model's KL regularisation loss
        if self.use_vocab_logits:
            log_probs = F.log_softmax(output.vocab_logits, dim=-1)
            log_init_probs = F.log_softmax(init_output.vocab_logits, dim=-1)
        else:
            log_probs = F.log_softmax(output.logits, dim=-1)
            log_init_probs = F.log_softmax(init_output.logits, dim=-1)

        kl_reg = F.kl_div(
            input=log_probs, 
            target=log_init_probs, 
            log_target=True,
            reduction = "batchmean"
        )

        # Get overall loss
        loss = ce_loss + self.alpha * kl_reg

        # Masking out all non-labels
        hits = torch.argmax(output.logits, dim=-1) == batch.labels
        acc = hits.sum()/len(batch.labels)

        #record training metrics
        self.record_metrics({
            'loss': loss.item(),
            'ce_loss':ce_loss.item(),
            'kl_reg':kl_reg.item(),
            'acc': acc.item(),
            'select': acc.item(),
        })

        return SimpleNamespace(
                    loss=loss, 
                    ce_loss=ce_loss,
                    kl_reg=kl_reg,
                    logits=output.logits, 
                    h=output.h
        )

    def to(self, device):
        self.init_model.to(device)
