# -*- coding: utf-8 -*-
# @Time    : 2022/7/16 15:39
# @Author  : sylviazz
# @FileName: metrics

class MRCScore(object):
    def __init__(self):
        # rel
        self.true_rel = .0
        self.pred_rel = .0
        self.gold_rel = .0
    def update_rel(self, pred_rel, gold_rel):
         self.gold_rel += len(gold_rel)
         self.pred_rel += len(pred_rel)
         for g in gold_rel:
            for p in pred_rel:
                if g == p:
                    self.true_rel += 1
         pre = 0 if self.true_rel + self.pred_rel == 0 else 1. * self.true_rel / self.pred_rel
         rec = 0 if self.true_rel + self.gold_rel == 0 else 1. * self.true_rel / self.gold_rel
         f1 = 0 if pre + rec == 0 else 2 * pre * rec / (pre + rec)
         return pre, rec, f1