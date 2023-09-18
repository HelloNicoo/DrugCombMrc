# -*- coding: utf-8 -*-
# @Time    : 2022/7/16 14:59
# @Author  : sylviazz
# @FileName: data_sample
class DataSample(object):
    def __init__(self,
                 text, target, row_id, drug_indices):
        self.text = text
        self.target = target
        self.row_id = row_id
        self.drug_indices = drug_indices