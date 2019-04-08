#!/usr/bin python3
# -*- coding: utf-8 -*-
from torch import optim

from config import VOCAB_DATA, TAG2ID_JSON, EMBEDDING_DATA
from config import HyperParams as hp
from preprocess import load_json_data, load_train_data, generate_batch_data
from lstm_crf import LSTMCRF


def run():
    import torch as th
    train_x, train_y, test_x, test_y = load_train_data()
    vocab = load_json_data(VOCAB_DATA)
    tag2id = load_json_data(TAG2ID_JSON)
    embedding = load_json_data(EMBEDDING_DATA)
    batch_size = hp.batch_size
    use_cuda = th.cuda.is_available()
    model = LSTMCRF(len(vocab), hp.embed_size, hp.num_units, len(tag2id))
    if use_cuda:
        model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=hp.lr, momentum=hp.momentum)

    def fit():
        for x_data, y_data in generate_batch_data(train_x, train_y, batch_size):
            optimizer.zero_grad()
            loss, _, _ = model(x_data)


if __name__ == '__main__':
    run()
