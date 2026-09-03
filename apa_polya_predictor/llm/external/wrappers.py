from __future__ import annotations

import torch
from torch import nn


class RNABertForSeqCls(nn.Module):
    def __init__(self, bert, hidden_size=120, class_num=2, freeze_backbone=False):
        super().__init__()
        self.bert = bert
        if freeze_backbone:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.classifier = nn.Linear(hidden_size, class_num)

    def forward(self, input_ids):
        _, pooled_output = self.bert(input_ids, attention_mask=input_ids > 0)
        return self.classifier(pooled_output)


class DNABERTForSeqCls(nn.Module):
    def __init__(self, model, pad_token_id=0, freeze_backbone=False):
        super().__init__()
        self.model = model
        self.pad_token_id = pad_token_id
        if freeze_backbone:
            base = getattr(self.model, "bert", self.model)
            for p in base.parameters():
                p.requires_grad = False

    def forward(self, input_ids):
        return self.model(input_ids, attention_mask=input_ids != self.pad_token_id).logits


class DNABERT2ForSeqCls(nn.Module):
    def __init__(self, model, pad_token_id=3, freeze_backbone=False):
        super().__init__()
        self.model = model
        self.pad_token_id = pad_token_id
        if freeze_backbone:
            base = getattr(self.model, "bert", self.model)
            for p in base.parameters():
                p.requires_grad = False

    def forward(self, input_ids):
        return self.model(input_ids, attention_mask=input_ids != self.pad_token_id).logits


class GENAForSeqCls(nn.Module):
    def __init__(self, model, pad_token_id=3, freeze_backbone=False):
        super().__init__()
        self.model = model
        self.pad_token_id = pad_token_id
        if freeze_backbone and hasattr(self.model, "bert"):
            for p in self.model.bert.parameters():
                p.requires_grad = False

    def forward(self, input_ids):
        return self.model(input_ids, attention_mask=input_ids != self.pad_token_id, output_hidden_states=False).logits


class UTRLMForSeqCls(nn.Module):
    def __init__(self, model, hidden_size=128, class_num=2, freeze_backbone=False):
        super().__init__()
        self.model = model
        if freeze_backbone and hasattr(self.model, "esm2"):
            for p in self.model.esm2.parameters():
                p.requires_grad = False
        self.classifier = nn.Linear(hidden_size, class_num)

    def forward(self, input_ids):
        x = self.model.forward_backbone(input_ids)[:, 0]
        return self.classifier(x)
