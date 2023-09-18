# -*- coding: utf-8 -*-
# @Time    : 2022/7/14 20:35
# @Author  : sylviazz
# @FileName: Model
from torch import nn
from transformers import BertModel, BertTokenizer
import torch
from DrugCombMrc.model import capsule

BertLayerNorm = torch.nn.LayerNorm
class BERTModel(nn.Module):
    def __init__(self, args, category_dim):
        super(BERTModel, self).__init__()
        # BERT模型
        self._tokenizer = BertTokenizer.from_pretrained(args.bert_path)
        special_tokens_dict = {'additional_special_tokens': ["[m]", "[/m]"]}
        self._tokenizer.add_special_tokens(special_tokens_dict)
        self._bert = BertModel.from_pretrained(args.bert_path)
        self._bert.resize_token_embeddings(len(self._tokenizer))
        self._bert1 = BertModel.from_pretrained(args.bert_path)
        self.classifier = nn.Linear(768, 3)
        self.hidden_size = 768
        # self.gdr1 = GatedDilatedResidualConv1D(self.hidden_size, 1)
        self.capsule_layer = capsule.Router(in_caps=1, out_caps=1, in_d=self.hidden_size, out_d=self.hidden_size, iterations = 5)
        self.weight = torch.rand(2)
        self.cap_linear = nn.Linear(2, 1)
        self.cls_linear = nn.Linear(768, 3)
        unfreeze_all_bert_layers: bool = False,
        unfreeze_final_bert_layer: bool = False,
        unfreeze_bias_terms_only: bool = True
        hidden_size = args.hidden_size
        self.weight = torch.softmax(self.weight, dim=0)
        hidden_size = args.hidden_size
        # for name, param in self._bert1.named_parameters():
        #     if unfreeze_final_bert_layer:
        #         if "encoder.layer.6" not in name:
        #             param.requires_grad = False
        #     elif unfreeze_bias_terms_only:
        #         if "bias" not in name:
        #             param.requires_grad = False
        #     elif not unfreeze_all_bert_layers:
        #         param.requires_grad = False
    def forward(self, para_embed, para_mask, add_text_embed, add_text_mask, context_query, context_query_mask):

        # hidden_states1 = self._bert(para_embed, attention_mask=para_mask)[0]
        # hidden_states2 = self._bert1(add_text_embed, attention_mask=add_text_mask)[0]
        # hidden_states1 = torch.max(hidden_states1, dim=1).values.unsqueeze(1)
        # hidden_states2 = torch.max(hidden_states2, dim=1).values.unsqueeze(1)
        # hidden_states = torch.cat((hidden_states1, hidden_states2), dim=1)
        # caps_output = self.capsule_layer(hidden_states)
        # # temp_tensor = torch.mul(temp_tensor1, temp_tensor2)
        # class_score = self.classifier(caps_output).squeeze(-1)
        # return class_score

        # 论文的模型
        # hidden_states1 = self._bert(para_embed, attention_mask=para_mask)[0]
        # hidden_states2 = self._bert1(add_text_embed, attention_mask=add_text_mask)[0]
        # new_hidden1 = hidden_states1[:, 0].unsqueeze(1)
        # new_hidden2 = hidden_states2[:, 0].unsqueeze(1)
        # hidden_states = torch.cat((new_hidden1, new_hidden2), dim=1)
        #
        # caps_output = self.capsule_layer(hidden_states)
        # temp = caps_output + new_hidden1
        # class_score = self.classifier(temp).squeeze()
        # return class_score
        # 消融query
        hidden_states1 = self._bert(para_embed, attention_mask=para_mask)[0]
        hidden_states2 = self._bert1(add_text_embed, attention_mask=add_text_mask)[0]
        new_hidden1 = hidden_states1[:, 0].unsqueeze(1)
        new_hidden2 = hidden_states2[:, 0].unsqueeze(1)
        # hidden_states = torch.cat((new_hidden1, new_hidden1), dim=1)

        caps_output = self.capsule_layer(new_hidden1)
        temp = caps_output
        class_score = self.classifier(temp).squeeze()
        return class_score
        # # cls直接分类
        # hidden_states1 = self._bert(para_embed, attention_mask=para_mask)[0]
        # # hidden_states2 = self._bert1(add_text_embed, attention_mask=add_text_mask)[0]
        # new_hidden1 = hidden_states1[:, 0]
        # # new_hidden2 = hidden_states2[:, 0]
        # # hidden_states = new_hidden2 + new_hidden1
        # class_score = self.cls_linear(new_hidden1).squeeze()
        # return class_score

        # 用pooling代替cls
        # hidden_states1 = self._bert(para_embed, attention_mask=para_mask)[0]
        # hidden_states2 = self._bert1(add_text_embed, attention_mask=add_text_mask)[0]
        # new_hidden1 = hidden_states1[:, :].max(1).values.unsqueeze(1)
        # new_hidden2 = hidden_states2[:, :].max(1).values.unsqueeze(1)
        # hidden_states = torch.cat((new_hidden1, new_hidden2), dim=1)
        # caps_output = self.capsule_layer(hidden_states)
        # temp = caps_output + new_hidden1
        # class_score = self.classifier(temp).squeeze()
        # return class_score

        # 去除胶囊网络
        # hidden_states1 = self._bert(para_embed, attention_mask=para_mask)[0]
        # hidden_states2 = self._bert1(add_text_embed, attention_mask=add_text_mask)[0]
        # new_hidden1 = hidden_states1[:, 0].unsqueeze(1)
        # new_hidden2 = hidden_states2[:, 0].unsqueeze(1)
        # hidden_states = torch.cat((new_hidden1, new_hidden2), dim=1)
        # hidden_states = hidden_states.permute(0, 2, 1)
        # caps_output = self.cap_linear(hidden_states)
        # caps_output = caps_output.permute(0, 2, 1)
        # temp = caps_output + new_hidden1
        # class_score = self.classifier(temp).squeeze()
        # return class_score
        #
        # hidden_states1 = self._bert(para_embed, attention_mask=para_mask)[0]
        # hidden_states2 = self._bert1(add_text_embed, attention_mask=add_text_mask)[0]
        # new_hidden1 = hidden_states1[:, 0].unsqueeze(1)
        # new_hidden2 = hidden_states2[:, 0].unsqueeze(1)
        # hidden_states = torch.cat((new_hidden1, new_hidden2), dim=1)
        # hidden_states = hidden_states.permute(0, 2, 1)
        # caps_output = self.cap_linear(hidden_states)
        # caps_output = caps_output.permute(0, 2, 1)
        # temp = caps_output + new_hidden1
        # class_score = self.classifier(temp).squeeze()
        # return class_score
        # 消融残差网络
        # hidden_states1 = self._bert(para_embed, attention_mask=para_mask)[0]
        # hidden_states2 = self._bert1(add_text_embed, attention_mask=add_text_mask)[0]
        # new_hidden1 = hidden_states1[:, 0].unsqueeze(1)
        # new_hidden2 = hidden_states2[:, 0].unsqueeze(1)
        # hidden_states = torch.cat((new_hidden1, new_hidden2), dim=1)
        # caps_output = self.capsule_layer(hidden_states)
        # temp = caps_output + new_hidden1
        # class_score = self.classifier(temp).squeeze()
        # return class_score