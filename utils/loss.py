# -*- coding: utf-8 -*-
# @Time    : 2022/7/16 15:36
# @Author  : sylviazz
# @FileName: loss
import torch
from torch.nn import functional as F
def normalize_size(tensor):
    if len(tensor.size()) == 3:
        tensor = tensor.contiguous().view(-1, tensor.size(2))
    elif len(tensor.size()) == 2:
        tensor = tensor.contiguous().view(-1)
    return tensor
def calculate_entity_loss(pred_start, pred_end, gold_start, gold_end):
    pred_start = normalize_size(pred_start)
    pred_end = normalize_size(pred_end)
    gold_start = normalize_size(gold_start)
    gold_end = normalize_size(gold_end)
    weight = torch.tensor([1, 3]).float().cuda()
    loss_start = F.cross_entropy(pred_start, gold_start.long(), reduction='sum', ignore_index=-1)
    loss_end = F.cross_entropy(pred_end, gold_end.long(), reduction='sum', ignore_index=-1)
    return 0.5 * loss_start + 0.5 * loss_end
def caculate_rel_loss(rel_score, target):
    target = torch.Tensor(target).cuda()
    loss = F.cross_entropy(rel_score, target.long(), reduction='sum', ignore_index=-1)
    return loss
