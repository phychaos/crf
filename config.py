#!/usr/bin python3
# -*- coding: utf-8 -*-

import os

TRAIN_PATH = os.path.join(os.getcwd(), 'data/train.char.bmes')
DEV_PATH = os.path.join(os.getcwd(), 'data/dev.char.bmes')
TEST_PATH = os.path.join(os.getcwd(), 'data/test.char.bmes')

PRE_TRAIN_DATA = os.path.join(os.getcwd(), 'data/gigaword_chn.all.a2b.uni.ite50.vec')
EMBEDDING_DATA = os.path.join(os.getcwd(), 'data/embedding.json')
VOCAB_DATA = os.path.join(os.getcwd(), 'data/vocab.json')

DATA_JSON = os.path.join(os.getcwd(), 'data/data.json')
TAG2ID_JSON = os.path.join(os.getcwd(), 'data/tag.json')


class HyperParams:
    batch_size = 128
    epoch = 50
    lr = 0.001
    num_units = 100
    embed_size = 50
    momentum = 0.5
