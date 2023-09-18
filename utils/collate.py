# -*- coding: utf-8 -*-
# @Time    : 2022/7/16 15:20
# @Author  : sylviazz
# @FileName: collate
import torch
from DrugCombMrc.example.tokenized_sample import TokenizedSample
def collate_fn(sample):
    target =  torch.tensor([s.target for s in sample], dtype=int)
    spos = [s.spos for s in sample]
    doc_id = [s.doc_id for s in sample]
    add_text_embed = torch.tensor([s.add_text_embed for s in sample], dtype=torch.long)
    add_text_mask = torch.tensor([s.add_text_mask for s in sample], dtype=torch.long)
    para_embed = torch.tensor([s.para_embed for s in sample], dtype=torch.long)
    para_mask = torch.tensor([s.para_mask for s in sample], dtype=torch.long)
    context_query = torch.tensor([s.context_query for s in sample], dtype=torch.long)
    context_query_mask = torch.tensor([s.context_query_mask for s in sample], dtype=torch.long)
    return TokenizedSample(doc_id, para_embed, para_mask, add_text_embed, add_text_mask,
                           context_query, context_query_mask, spos, target)
