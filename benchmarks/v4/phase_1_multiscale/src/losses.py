from __future__ import annotations

import torch
from .metrics import soft_dice_loss


class SegmentationLoss(torch.nn.Module):
    def __init__(self,num_classes:int,ignore:int,background:set[int],ce_weight:float,dice_weight:float,class_weights:torch.Tensor|None=None):
        super().__init__(); self.ce=torch.nn.CrossEntropyLoss(weight=class_weights,ignore_index=ignore); self.n,self.ignore,self.background,self.a,self.b=num_classes,ignore,background,ce_weight,dice_weight
    def forward(self,logits:torch.Tensor,target:torch.Tensor)->tuple[torch.Tensor,dict[str,float]]:
        ce=self.ce(logits,target); dice=soft_dice_loss(logits,target,self.n,self.ignore,self.background); total=self.a*ce+self.b*dice
        return total,{"ce":float(ce.detach()),"dice_loss":float(dice.detach()),"loss":float(total.detach())}
