# -*- coding: utf-8 -*-
# @Time    : 2022/7/14 19:46
# @Author  : sylviazz
# @FileName: main.py
import json
import os
import sys
import time

from utils.convert2final import filter_overloaded_predictions, write_jsonl
from utils.eval import f_score

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import datetime
from torch.optim import AdamW
from utils.args import init_args
from utils.processor import init_logger
from utils.processor import seed_everything
from utils.labels import get_entity_category
from utils.collate import collate_fn
from model.Model import BERTModel
from utils.dataset import DCDataset
from train import Trainer
import torch
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
args = init_args()

# ##########log##########
# log_file:日志保存的路径 output/logs
log_dir = args.output_dir + '/logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file = "./output/logs/{}.log".format(time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()))
logger = init_logger(log_file=log_file)


# ##########seed##########
seed_everything(args.seed)

logger.info("Building drug_combo model...")
category_list = get_entity_category(args.task, args.datatype)
model = BERTModel(args, len(category_list[0]))
model = model.cuda()
class Logger(object):
    def __init__(self, fileN="default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
log_path = './Logs/'
if not os.path.exists(log_path):
    os.makedirs(log_path)
# 日志文件名按照程序运行时间设置
log_file_name = log_path + 'log-' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.log'
# 记录正常的 print 信息
sys.stdout = Logger(log_file_name)
# 记录 traceback 异常信息
sys.stderr = Logger(log_file_name)
if args.do_train:
    tokenizer = BertTokenizer.from_pretrained(args.bert_path, local_files_only=False)
    special_tokens_dict = {'additional_special_tokens': ["[m]", "[/m]"]}
    tokenizer.add_special_tokens(special_tokens_dict)
    train_dataset = DCDataset(tokenizer, args.data_path, args.bert_path, "final_train_set", args.task, 'train',args.max_len)
    test_dataset = DCDataset(tokenizer, args.data_path, args.bert_path, "final_test_set", args.task, 'test',args.max_len)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True,collate_fn=collate_fn)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, collate_fn=collate_fn)

    # optimizer
    logger.info('initial optimizer......')
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if "_bert" in n], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if "_bert" not in n],
         'lr': args.learning_rate1, 'weight_decay': 0.01}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate2)

    # scheduler
    batch_num_train = len(train_dataset) // args.train_batch_size
    training_steps = args.num_train_epochs * batch_num_train
    warmup_steps = int(training_steps * args.warm_up)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=training_steps)

    # ##########Training##########
    # model.load_state_dict(torch.load('save_modelner/model.pth')['net'])
    trainer = Trainer(logger, model, optimizer, scheduler, args, category_list)
    best_eval_f1 = 0
    best_test_f1 = 0
    for epoch in range(args.num_train_epochs):
        epoch_start_time = time.time()
        logger.info('+' * 102)
        msg = "第" + str(epoch + 1) + "个epoch"
        logger.info(msg)
        # trainer.train(train_dataloader, epoch)
        p, rec, results = trainer.eval(test_dataloader, epoch)
        sys.stdout = Logger('datalog.txt')
        logger.info((p, rec, results))
        pred_file_name = os.path.join('epoch_{}'.format(epoch))
        gold_file_name = "utils/temp_out.jsonl"
        final_data = []
        with open(pred_file_name, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                temp_dict = {}
                data = json.loads(line)
                temp_dict['doc_id'] = data['doc_id']
                temp_dict['drug_idxs'] = data['drug_idx']
                temp_dict['relation_label'] = data['pred']
                final_data.append(temp_dict)
        fixed_test = filter_overloaded_predictions(final_data)
        os.makedirs("outputs", exist_ok=True)
        pred_output = os.path.join("outputs", "final_predictions.jsonl")
        write_jsonl(fixed_test, pred_output)
        with open(pred_output) as f:
            pred = [json.loads(l) for l in f.readlines()]
        with open(gold_file_name) as f:
            gold = [json.loads(l) for l in f.readlines()]
        if (p == 0 or rec == 0.0 or results == 0):
            print("未收敛")
            f_partial = 0
        else:
            f_partial, p_partial, r_partial = f_score(gold, pred, exact_match=False, any_comb=True)
            f_labeled_partial, p_labeled_partial, r_labeled_partial = f_score(gold, pred, exact_match=False,
                                                                              any_comb=False)
            f_exact, p_exact, r_exact = f_score(gold, pred, exact_match=True, any_comb=True)
            f_labeled_exact, p_labeled_exact, r_labeled_exact = f_score(gold, pred, exact_match=True, any_comb=False)
            logger.info(("f_partial: " + str(f_partial) + " f_labeled_partial: " + str(f_labeled_partial) +
                  " f_exact: " + str(f_exact) + " f_labeled_exact: " + str(f_labeled_exact)))
        if (f_partial > best_eval_f1):
            best_eval_f1 = results
            state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            save_path = args.save_model_path + args.task + '/' + args.datatype + '/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(state, save_path + 'model.pth')
    """
        将实验结果发送至邮箱
    """
    # 设置smtp所需要的参数
    smtp_server = "smtp.qq.com"  # 邮箱服务器
    username = "371209945@qq.com"  # 账号
    password = "qtfmpfeizyvocbac"  # 授权码
    sender = "371209945@qq.com"  # 发邮件的人
    receiver = ['371209945@qq.com']  # 2个收邮件的人
    subject = "实验结果"  # 邮件主题

    # 构造邮件对象MIMEMultipart
    # 主题、发件人、收件人、日期显示在邮件页面上
    msg = MIMEMultipart('mixed')
    msg['Subject'] = subject
    msg['From'] = '371209945@qq.com <371209945@qq.com>'
    msg['TO'] = ";".join(receiver)
    msg['Date'] = '2022.11.10'

    # 构造文字内容
    current_time = datetime.datetime.now()
    text = str(current_time) + " 以下是实验结果, 最好结果是：" + str(best_eval_f1)
    text_plain = MIMEText(text, 'plain', 'utf-8')
    msg.attach(text_plain)

    # 定义要读取的文件夹路径
    folder_path = "output/logs"
    folder_list = []

    # 遍历文件夹
    for root, dirs, files in os.walk(folder_path):
        folder_list.append(files)
    name_list = folder_list[0]
    name_list = sorted(name_list)
    last_path = folder_path + '/' + name_list[-1]

    # 构造附件
    send_file = open(last_path, 'rb').read()
    text_att = MIMEText(send_file, 'base64', 'utf-8')
    text_att["Content-Type"] = 'application/octet-stream'

    # 重命名附件文件
    text_att.add_header('Content-Disposition', 'attachment', filename='测试.txt')
    msg.attach(text_att)

    # 发送邮件
    s = smtplib.SMTP()
    s.connect(smtp_server, 25)
    s.login(username, password)
    s.sendmail(sender, receiver, msg.as_string())
    s.quit()