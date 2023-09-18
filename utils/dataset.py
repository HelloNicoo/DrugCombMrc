# -*- coding: utf-8 -*-
# @Time    : 2022/7/14 20:46
# @Author  : sylviazz
# @FileName: dataset

from torch.utils.data import Dataset
from itertools import chain, combinations
from transformers import BertTokenizer
import numpy as np
from example.data_sample import DataSample
from example.tokenized_sample import TokenizedSample
from utils.labels import get_entity_category, get_rel_type
from utils.processor import get_para, read_jsonl, find_index, get_para_drug_pos, get_para_test
def find_pos(sentence, para):
    sen_len = len(sentence)
    par_len = len(para)
    for i in range(par_len):
        temp = para[i: sen_len + 1]
        if(temp == sentence):
            return i
    return -1
class DCDataset(Dataset):
    def __init__(self, tokenizer, data_path, model_path, dataset_type, task, data_type, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.total_rel = []
        self._build_examples(data_path, dataset_type, task, data_type)

# ======================进行tokenizer处理======================
    def __getitem__(self, item):
        rel2id = {'POS': 2, 'COMB': 1, 'NEG': 1, 'NOT_COMB': 0}
        sample = self.total_rel[item]
        sentence = sample['sentence']
        para = sample['para']
        spos = sample['spos']
        target = sample['target']
        doc_id = sample['doc_id']
        add_text = sample['add_text'] + '?'
        para_content = para.split(' ')
        context_window_size = 450
        first_entity_start_token = min([i for i, t in enumerate(para_content) if t == "[m]"])
        final_entity_end_token = max([i for i, t in enumerate(para_content) if t == "[/m]"])
        entity_distance = final_entity_end_token - first_entity_start_token
        add_left = (context_window_size - entity_distance) // 2
        start_window_left = max(0, first_entity_start_token - add_left)
        add_right = (context_window_size - entity_distance) - add_left
        start_window_right = min(len(para_content), final_entity_end_token + add_right)
        para = para_content[start_window_left: start_window_right]
        para = ' '.join(para)
        para_emb = self.tokenizer.tokenize(para)
        add_text = self.tokenizer.tokenize(add_text)
        new_para = ['[CLS]'] + para_emb[: 480]
        new_add_text = ['[CLS]'] + add_text



        para_embed = self.tokenizer.convert_tokens_to_ids(new_para)
        add_text_embed = self.tokenizer.convert_tokens_to_ids(new_add_text)

        # 单编码器
        context_query = ['[CLS]'] + para_emb[: 450] + ['[SEP]'] + add_text
        context_query_embed = self.tokenizer.convert_tokens_to_ids(context_query)
        context_query_mask = []
        context_query_mask.extend([1] * len(context_query_embed))
        context_query_len = 512 - len(context_query_embed)
        context_query_embed.extend([0] * context_query_len)
        context_query_mask.extend([0] * context_query_len)

        para_mask = []
        para_mask.extend([1] * len(para_embed))
        add_text_mask = []
        add_text_mask.extend([1] * len(add_text_embed))
        para_pad_len = 512 - len(para_embed)
        add_context_pad_len = 512 - len(add_text_mask)

        para_embed.extend([0] * para_pad_len)
        para_mask.extend([0] * para_pad_len)

        add_text_embed.extend([0] * add_context_pad_len)
        add_text_mask.extend([0] * add_context_pad_len)
        # 单编码器
        return TokenizedSample(doc_id, para_embed, para_mask, add_text_embed, add_text_mask,
                               context_query_embed, context_query_mask, spos, target)

        # return TokenizedSample(doc_id, para_embed, para_mask, add_text_embed, add_text_mask, spos, target)20
    def __len__(self):
        return len(self.total_rel)
    def _build_examples(self, data_path, dataset_type, task, data_type):
        a = data_path + '/' + dataset_type + '.jsonl'
        lines = read_jsonl(a)
        sentence_list, para_list, spos_list, add_text_list, target_list, doc_list = get_para(lines)
        for k in range(len(sentence_list)):
            relations = []
            relations.append({'sentence':sentence_list[k], 'para':para_list[k], 'spos': spos_list[k], \
                                'add_text':add_text_list[k], 'doc_id':doc_list[k]
                                , 'target': target_list[k]})
            self.total_rel.extend(relations)