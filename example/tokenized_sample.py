# -*- coding: utf-8 -*-
# @Time    : 2022/7/16 15:20
# @Author  : sylviazz
# @FileName: tokenized_sample
# class TokenizedSample(object):
#     def __init__(self,
#                  doc_id, para_embed, para_mask, add_text_embed, add_text_mask, spos, target):
#         self.doc_id = doc_id
#         self.para_embed = para_embed
#         self.para_mask = para_mask
#         self.add_text_embed = add_text_embed
#         self.add_text_mask = add_text_mask
#         self.spos = spos
#         self.target = target
#单编码器
class TokenizedSample(object):
    def __init__(self,
                 doc_id, para_embed, para_mask, add_text_embed, add_text_mask,
                 context_query, context_query_mask, spos, target):
        self.doc_id = doc_id
        self.para_embed = para_embed
        self.para_mask = para_mask
        self.add_text_embed = add_text_embed
        self.add_text_mask = add_text_mask
        self.context_query = context_query
        self.context_query_mask = context_query_mask
        self.spos = spos
        self.target = target