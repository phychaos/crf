#!/usr/bin python3
# -*- coding: utf-8 -*-
from config import *
import numpy as np
import json

PAD = '<pad>'
UNK = '<unk>'
PAD2ID = 0
UNK2ID = 1


def load_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as fp:
        for line in fp.readlines():
            data.append(line.strip())
    return data


def save_json_data(data, filename):
    with open(filename, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, ensure_ascii=False)


def load_json_data(filename):
    with open(filename, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return data


def read_label_data(filename):
    """
    加载数据集
    :param filename:
    :return:
    """
    text = []
    tags = []
    lines = load_data(filename)
    sentence = []
    tag = []
    for line in lines:
        token = line.split(' ')
        if len(token) == 2:
            sentence.append(token[0])
            tag.append(token[1])
        elif len(token) == 1 and len(sentence) > 0:
            text.append(sentence)
            tags.append(tag)
            sentence = []
            tag = []
    if len(sentence) > 0:
        text.append(sentence)
        tags.append(tag)
    return text, tags


def build_vocab(filename, embed_size=50):
    """
    构建词典词向量
    :param filename:
    :param embed_size:
    :return:
    """
    word2id = {PAD: PAD2ID, UNK: UNK2ID}

    pad2vec = [0] * embed_size
    unk2vec = [float(ii) for ii in np.random.random(embed_size)]
    embedding = [pad2vec, unk2vec]
    embed_data = load_data(filename)
    for line in embed_data:
        token = line.split(' ')
        if len(token) != embed_size + 1:
            continue
        word = token[0]
        word_vec = [float(_) for _ in token[1:]]
        if word in word2id:
            continue
        word2id[word] = len(word2id)
        embedding.append(word_vec)
    save_json_data(word2id, VOCAB_DATA)
    save_json_data(embedding, EMBEDDING_DATA)


def preprocess():
    """
    数据预处理
    :return:
    """
    train_x, train_y = read_label_data(TRAIN_PATH)
    test_x, test_y = read_label_data(TEST_PATH)
    dev_x, dev_y = read_label_data(DEV_PATH)
    word2id = load_json_data(VOCAB_DATA)
    tags = set()
    for tag in train_y:
        tags.update(tag)
    tag2id = {tag: idx for idx, tag in enumerate(tags)}
    train_x, train_y = text2idx(train_x, word2id), text2idx(train_y, tag2id, 0)
    test_x, test_y = text2idx(test_x, word2id), text2idx(test_y, tag2id, 0)
    dev_x, dev_y = text2idx(dev_x, word2id), text2idx(dev_y, tag2id, 0)
    data = {
        "train_x": train_x, "train_y": train_y,
        "test_x": test_x, "test_y": test_y,
        "dev_x": dev_x, "dev_y": dev_y,
    }
    save_json_data(data, DATA_JSON)
    save_json_data(tag2id, TAG2ID_JSON)


def text2idx(data, data2id, default=1):
    """
    字符转id
    :param data:
    :param data2id:
    :param default:
    :return:
    """
    _data = []
    for seq in data:
        seq2id = [data2id.get(kk, default) for kk in seq]
        _data.append(seq2id)
    return _data


def generate_batch_data(x, y, batch_size, use_cuda=False):
    import torch as th
    from torch.autograd import Variable
    sample_size = len(y)
    total_epoch = sample_size // batch_size

    for ii in range(total_epoch):
        start, end = ii * batch_size, (ii + 1) * batch_size
        if start >= sample_size:
            continue
        if end >= sample_size:
            end = sample_size
        x_data, y_data = x[start:end], y[start:end]
        x_data = th.Tensor(padding_sentence(x_data))
        y_data = th.Tensor(padding_sentence(y_data))
        if use_cuda:
            x_data, y_data = x_data.cuda(), y_data.cuda()
        x_data, y_data = Variable(x_data).long(), Variable(y_data).long()
        yield x_data, y_data


def padding_sentence(data, default=0):
    max_len = max([len(kk) for kk in data])
    for kk, sentence in enumerate(data):
        seq_len = len(sentence)
        if seq_len >= max_len:
            sentence = sentence[:max_len]
        else:
            sentence += [default] * (max_len - seq_len)
        data[kk] = sentence
    return data


def load_train_data():
    data = load_json_data(DATA_JSON)
    train_x = data['train_x']
    train_y = data['train_y']
    test_x = data['test_x']
    test_y = data['test_y']
    return train_x, train_y, test_x, test_y



if __name__ == '__main__':
    build_vocab(PRE_TRAIN_DATA, embed_size=50)
    preprocess()
