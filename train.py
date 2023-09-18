# -*- coding: utf-8 -*-
# @Time    : 2022/7/16 15:34
# @Author  : sylviazz
# @FileName: train
import os

import torch
from tqdm import tqdm
from transformers import BertTokenizer

from utils.loss import calculate_entity_loss, caculate_rel_loss
from utils.metric import DRScore
from torch.nn import functional as F
from torch import nn
class SupConLoss(nn.Module):
    """https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-30)

        # compute mean of log-likelihood over positive
        # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # 如果出现loss为Nan的情况，则需要加一个特别小的数 防止除零
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
def calculate_SCL_loss(gold, pred_scores):
    SCL = SupConLoss(contrast_mode='all', temperature=0.9)
    gold = torch.tensor(gold)
    answer = gold
    idxs = torch.nonzero(answer != -1).squeeze()

    answers, score_list = [], []
    for i in idxs:
        answers.append(answer[i])
        score_list.append(pred_scores[i])

    # label 维度变为:[trans_dim]
    answers = torch.stack(answers)
    # 进行维度重构 category维度:[category_nums, 1, trans_dim]
    scores = torch.stack(score_list)

    scores = F.softmax(scores, dim=1)
    scores = scores.unsqueeze(1)

    scl_loss = SCL(scores, answers)
    scl_loss = scl_loss / len(scores)

    return scl_loss
class Trainer:
    def __init__(self, logger, model, optimizer, scheduler, args, category_list):
        self.logger = logger
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_path)
        self.category_list = category_list

    def train(self, train_dataloader, epoch):
        print("***** Running training *****")
        self.model.train()
        self.model.zero_grad()
        with tqdm(total=len(train_dataloader), desc="train") as pbar:
            for batch_idx, batch in enumerate(train_dataloader):
                self.optimizer.zero_grad()
                # q1_forward
                combo_class = self.model(batch.para_embed.cuda(), batch.para_mask.cuda(),\
                                         batch.add_text_embed.cuda(), batch.add_text_mask.cuda(),
                                         batch.context_query.cuda(), batch.context_query_mask.cuda()
                                         )
                loss = caculate_rel_loss(combo_class.squeeze(), batch.target.cuda())

                scl_loss = calculate_SCL_loss(batch.target, combo_class.squeeze())
                total_loss = (1 - self.args.contrastive_lr) * loss + self.args.contrastive_lr * scl_loss
                total_loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                pbar.set_description(f'Epoch [{epoch + 1}/{self.args.num_train_epochs}]')
                pbar.set_postfix({'loss': '{0:1.5f}'.format(loss)})
                pbar.update(1)

    def eval(self, eval_dataloader, epoch):
        print("***** Running evaluation epoch {} *****".format(epoch))
        with open(os.path.join('true_pred}'), 'a', encoding='utf-8') as f:
            f.write(str(epoch) + '\n')
        mrc_score = DRScore()
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(eval_dataloader):
                class_score = self.model(batch.para_embed.cuda(), batch.para_mask.cuda(),\
                                         batch.add_text_embed.cuda(), batch.add_text_mask.cuda(),
                                         batch.context_query.cuda(), batch.context_query_mask.cuda()
                                         ).squeeze(-1)
                class_score = F.softmax(class_score, dim=-1)
                # temp = class_score[0]
                temp = class_score
                _, class_result = torch.max(temp, dim=0)
                class_result = class_result.item()
                p, rec, f1 = mrc_score.update_rel(class_result, batch.target[0].item())
                result_dict = {}
                result_dict["doc_id"] = batch.doc_id
                result_dict["drug_idx"] = batch.spos[0]
                result_dict["pred"] = class_result
                if ((class_result == batch.target[0]) and (class_result != 0) and (batch.target[0] != 0)):
                    with open(os.path.join('true_pred}'), 'a', encoding='utf-8') as f:
                        f.write(str(result_dict) + '\n')
                with open(os.path.join('epoch_{}'.format(epoch)), 'a', encoding='utf-8') as f:
                    f.write(str(result_dict).replace("'",'"') + '\n')
                if f1 == None:
                    f1 = 0
                if p == None:
                    p = 0
                if rec == None:
                    rec = 0
        return p, rec, f1

