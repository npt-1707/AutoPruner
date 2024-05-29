import pandas as pd
import logging
import os
from sklearn.metrics import confusion_matrix
import math
import json
import torch
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def read_config_file(config_file):
    configs = {}
    with open(config_file) as f:
        for line in f.readlines():
            line = line.strip()
            split_line = line.split("=")
            configs[split_line[0]] = split_line[1]
    return configs


def get_input_and_mask(src, dst , max_length, tokenizer):
    src_tokens = tokenizer.tokenize(src)
    dst_tokens = tokenizer.tokenize(dst)
    
    has_cls_and_sep = ((tokenizer.cls_token is not None) and (tokenizer.sep_token is not None))
    
    if has_cls_and_sep:
        tokens=[tokenizer.cls_token]+src_tokens+[tokenizer.sep_token]+dst_tokens+[tokenizer.sep_token]
    else:
        tokens = src_tokens + dst_tokens

    token_length = len(tokens)
    if  token_length > max_length:
        truncation_ratio = max_length/token_length
        num_special_tokens = 3 if has_cls_and_sep else 0
        src_len = len(src_tokens)
        dst_len = len(dst_tokens)
        if  src_len < dst_len:
            src_tokens = src_tokens[:int(len(src_tokens) * truncation_ratio)]
            dst_tokens = dst_tokens[:max_length - len(src_tokens) - num_special_tokens]
        else:
            dst_tokens = dst_tokens[:int(len(dst_tokens) * truncation_ratio)]
            src_tokens = src_tokens[:max_length - len(dst_tokens) - num_special_tokens]
        if has_cls_and_sep:
            new_tokens=[tokenizer.cls_token]+src_tokens+[tokenizer.sep_token]+dst_tokens+[tokenizer.sep_token]
        else:
            new_tokens=src_tokens+dst_tokens
        mask = [1 for _ in range(len(new_tokens))]
    else:
        new_tokens = [tokens[i] if i < token_length else tokenizer.pad_token for i in range(max_length)]
        mask = [1 if i < token_length else 0 for i in range(max_length)]

    tokens_ids= tokenizer.convert_tokens_to_ids(new_tokens)
    if len(tokens_ids) > max_length:
        print(len(dst_tokens))
        print(len(src_tokens))
        print(len(tokens_ids))
        import pdb
        pdb.set_trace()
        raise "Truncation errors"
    return tokens_ids, mask

def load_code(path):
    data = {}
    df = pd.read_csv(path)
    descriptor = df['descriptor']
    code = df['code']
    for i in range(len(descriptor)):
        if isinstance(code[i], str):
            data[descriptor[i]] = " ".join(code[i].replace("\n", " ").split())
    return data


import logging


class Logger:
    def __init__(
        self, log_file, log_level=logging.INFO, console=True, file_mode="w", time=True
    ):
        self.logger = logging.getLogger()
        self.logger.setLevel(log_level)

        if time:
            formatter = logging.Formatter(
                "[%(asctime)s] - [%(levelname)s]: %(message)s"
            )
        else:
            formatter = logging.Formatter("[%(levelname)s]: %(message)s")

        # Create a file handler
        file_handler = logging.FileHandler(log_file, mode=file_mode)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Create a console handler
        if console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def log(self, message, level=logging.INFO):
        self.logger.log(level, message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)


class AverageMeter(object):
    '''Computes and stores the average and current value'''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum/self.count

    def item(self):
        return self.avg

def evaluation_metrics(label, pred, cfx_matrix):
    cfx_matrix += confusion_matrix(label, pred, labels = [0, 1])
    (tn, fp), (fn, tp) = cfx_matrix
    precision = tp/(tp + fp)
    if math.isnan(precision):
        precision = 0
    
    recall = tp/(tp + fn)
    
    if precision + recall != 0:
        f1 = 2*precision*recall/(precision+recall)
    else:
        f1 = 0
    
    f1 = 2*precision*recall/(precision + recall)
    return cfx_matrix, precision, recall, f1

def load_json(file):
    if os.path.exists(file):
        with open(file, "r") as f:
            return json.load(f)
    return {}

def save_json(data, file):
    with open(file, "w") as f:
        json.dump(data, f)