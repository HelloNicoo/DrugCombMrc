# -*- coding: utf-8 -*-
# @Time    : 2022/7/14 19:55
# @Author  : sylviazz
# @FileName: processor

import logging
import os
import random
from pathlib import Path
import json
import numpy as np
import torch
def find_drug_pos(para, drug):
    para_len = len(para)
    drug_len = len(drug)
    drug_pos = []
    for i in range(para_len):
        if para[i: drug_len + 1] == drug:
            drug_pos.append(i)
    return drug_pos
def init_logger(log_file=None, log_file_level=logging.NOTSET):
    if isinstance(log_file, Path):
        log_file = str(log_file)
    log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                   datefmt='%m/%d/%Y %H:%M:%S')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)  #定义控制台等级
        # file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)   #文本的处理器加控制台处理器
    return logger


def seed_everything(seed=1024):
    """
    设置整个开发环境的seed
    :param seed:
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
def to_text(str):
    text_list = []
    for i in range(len(str)):
        text_list.append(str[i])
    return text_list
def get_para(lines):
    sentence_list = []
    para_list = []
    spos_list = []
    add_text_list = []
    target_list = []
    doc_list = []
    for line in lines:
        line = json.loads(line)
        sentence, para, spos, add_text, target, doc_id = line['text'], line['doc'], line['pow'], line['add_text'], line['target'], line['doc_id']
        sentence_list.append(sentence)
        para_list.append(para)
        spos_list.append(spos)
        add_text_list.append(add_text)
        target_list.append(target)
        doc_list.append(doc_id)
    return sentence_list, para_list, spos_list, add_text_list, target_list, doc_list
def get_para_test(lines):
    sentence_list = []
    para_list = []
    spos_list = []
    add_text_list = []
    target_list = []
    doc_list = []
    num = 0
    for line in lines:
        num += 1
        print(num)
        line = json.loads(line)
        sentence, para, spos, add_text, doc_id = line['text'], line['doc'], line['pow'], line['add_text'], line['doc_id']
        sentence_list.append(sentence)
        para_list.append(para)
        spos_list.append(spos)
        add_text_list.append(add_text)
        doc_list.append(doc_id)
    return sentence_list, para_list, spos_list, add_text_list, doc_list
def read_jsonl(data_path):
    with open(data_path, 'r', encoding='utf-8') as fp:
        lines = fp.read().splitlines()
    fp.close()
    return lines
def find_index(para_content, item):
    temp = []
    item = item.split(' ')
    len_item = len(item)
    for i in range(len(para_content)):
        if(para_content[i: i + len_item] == item):
            temp.append((i, i + len_item))
    return temp
def get_para_drug_pos(para, drug_token, sentence):
    para = para.split(' ')
    sentence = sentence.split(' ')
    len_para = len(para)
    len_sentence = len(sentence)
    sen_pos = 0
    for i in range(len_para):
        if(para[i:i + len_sentence] == sentence):
            sen_pos = i
            break
    new_drug_token = []
    for i in range(len(drug_token)):
        temp = (drug_token[i][0] + sen_pos, drug_token[i][1] + sen_pos)
        new_drug_token.append(temp)
    return new_drug_token
