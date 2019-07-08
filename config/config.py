#!/usr/bin python3
# -*- coding: utf-8 -*-

import os

# TRAIN_PATH = os.path.join(os.getcwd(), 'data/train.char.bmes')
TRAIN_PATH = os.path.join(os.getcwd(), 'data/train_ner.data')
DEV_PATH = os.path.join(os.getcwd(), 'data/test.char.bmes')
# TEST_PATH = os.path.join(os.getcwd(), 'data/test.char.bmes')
TEST_PATH = os.path.join(os.getcwd(), 'data/test_ner.data')

PRE_TRAIN_DATA = os.path.join(os.getcwd(), 'data/gigaword_chn.all.a2b.uni.ite50.vec')
EMBEDDING_DATA = os.path.join(os.getcwd(), 'data/embedding.json')
VOCAB_DATA = os.path.join(os.getcwd(), 'data/vocab.json')
chinese_vocab = os.path.join(os.getcwd(), 'bert_model/vocab.txt')

DATA_JSON = os.path.join(os.getcwd(), 'data/data.json')
BERT_DATA_JSON = os.path.join(os.getcwd(), 'data/bert_data.json')
TAG2ID_JSON = os.path.join(os.getcwd(), 'data/tag.json')
BERT_TAG2ID_JSON = os.path.join(os.getcwd(), 'data/bert_tag.json')
BERT_PRETAIN_PATH = os.path.join(os.getcwd(), 'bert_model')
ELMO_PRETAIN_PATH = os.path.join(os.getcwd(), 'elmo_model')
ELMO_DATA_JSON = os.path.join(os.getcwd(), 'data/elmo_data.json')
ELMO_TAG2ID_JSON = os.path.join(os.getcwd(), 'data/elmo_tag.json')
MODEL_PATH = os.path.join(os.getcwd(), 'data/model_bert.pkl')
epochs = 60


class HyperParams:
	num_layers = 2
	batch_size = 64
	lr = 5e-4
	num_units = 150
	embed_size = 200
	label_num_units = 100
	topk = 25
	lr_decay = 0.9


class BertParams:
	num_layers = 1
	batch_size = 16
	lr = 5e-5
	num_units = 768
	rnn_hidden = 200
	lr_decay = 0.9


class ElmoParams:
	num_layers = 1
	batch_size = 32
	lr = 5e-3
	num_units = 1024
	rnn_hidden = 200
	lr_decay = 0.9
