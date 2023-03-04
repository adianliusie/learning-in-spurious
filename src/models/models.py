import torch
import torch.nn as nn

from types import SimpleNamespace
from transformers import logging

from .pre_trained_trans import load_transformer, load_MLM_transformer
from .tokenizers import load_tokenizer

logging.set_verbosity_error()

class TransformerModel(torch.nn.Module):
    """basic transformer model for multi-class classification""" 
    def __init__(
        self, 
        trans_name:str, 
        num_classes:int=2
    ):
        super().__init__()
        self.transformer = load_transformer(trans_name)
        h_size = self.transformer.config.hidden_size
        self.output_head = nn.Linear(h_size, num_classes)
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        **kwargs
    ):
        # get transformer hidden representations
        trans_output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # get CLS hidden vector and convert to logits through classifier
        H = trans_output.last_hidden_state  #[bsz, L, 768] 
        h = H[:, 0]                         #[bsz, 768] 
        logits = self.output_head(h)             #[bsz, C] 
        return SimpleNamespace(
            h=h, 
            logits=logits
        )

    def freeze_transformer(self):
        for param in self.transformer.parameters():
            param.requires_grad = False

    def unfreeze_transformer(self):
        for param in self.transformer.parameters():
            param.requires_grad = True
    
    def freeze_classifier_bias(self):
        self.output_head.bias.requires_grad = False

class PromptFinetuning(torch.nn.Module):
    def __init__(
            self, 
            trans_name:str, 
            label_words:list, 
    ):
        super().__init__()
        self.transformer = load_MLM_transformer(trans_name)
        self.tokenizer   = load_tokenizer(trans_name)
        self.label_ids   = [self.tokenizer(word).input_ids[1] for word in label_words]
        
    def forward(
        self,
        input_ids,
        attention_mask=None,
        mask_positions=None,
    ):
        # encode everything and get MLM probabilities
        trans_output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # select MLM probs of the masked positions, only for the label ids
        mask_pos_logits = trans_output.logits[torch.arange(input_ids.size(0)), mask_positions]
        logits = mask_pos_logits[:, tuple(self.label_ids)]
        h = trans_output.hidden_states[-1][torch.arange(input_ids.size(0)), mask_positions]

        return SimpleNamespace(
            h = h, 
            logits=logits,
            vocab_logits=mask_pos_logits
        )
        
    def update_label_words(self, label_words:str):
        self.label_ids = [self.tokenizer(word).input_ids[1] for word in label_words]

    def freeze_head(self):
        for param in self.transformer.lm_head.parameters():
            param.requires_grad = False

