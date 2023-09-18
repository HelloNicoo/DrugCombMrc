# -*- coding: utf-8 -*-
# @Time    : 2022/7/14 19:48
# @Author  : sylviazz
# @FileName: args.py
import argparse

def init_args():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--data_path", default='./new_data', type=str, required=False, help="The path of data")
    parser.add_argument("--task", default='ner', type=str, required=False,
                        help="The name of the task, selected from: [re, ner]")
    parser.add_argument("--datatype", default='', type=str, required=False,
                        help="The type of the dataset, selected from: [laptop, rest], [rest15, rest16]")
    parser.add_argument("--bert_path", default='pubmedbert',
                        type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--output_dir", default='./output', type=str, help="The dir of output")
    parser.add_argument("--save_model_path", default='./save_model', type=str, help="The dir of save_model")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev/test set.")

    # Other parameters
    parser.add_argument("--max_len", default=512, type=int)
    parser.add_argument("--n_gpu", default=2)
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=1, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate1", default=2e-4, type=float)
    parser.add_argument("--learning_rate2", default=1e-5, type=float)
    parser.add_argument("--contrastive_lr", default=1e-5, type=float)
    parser.add_argument("--num_train_epochs", default=10, type=int, help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=2021, help="random seed for initialization")
    parser.add_argument('--warm_up', type=float, default=0.1)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--beta', type=float, default=0.8)
    args = parser.parse_args()

    return args
