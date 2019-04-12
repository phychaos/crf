#!/usr/bin python3
# -*- coding: utf-8 -*-
from config.config import *
import numpy as np
import json
import torch as th
import opencc

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


def read_label_data(filename, split=' '):
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
		token = line.split(split)
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


def preprocess_bert():
	train_x, train_y = read_label_data(TRAIN_PATH)
	test_x, test_y = read_label_data(TEST_PATH)
	labels = set()
	for label in train_y:
		labels.update(label)
	labels.update({'X', '[CLS]', '[SEP]'})
	labels = list(labels)
	tag2id = {label: idx for idx, label in enumerate(labels)}
	data = {
		"train_x": train_x, "train_y": train_y,
		"test_x": test_x, "test_y": test_y,
	}
	save_json_data(data, BERT_DATA_JSON)
	save_json_data(tag2id, BERT_TAG2ID_JSON)


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


class TokenizerBertText(object):
	def __init__(self, use_cuda, pre_process=False):
		from pytorch_pretrained_bert import BertTokenizer
		self.use_cuda = use_cuda
		self.tokenizer = BertTokenizer.from_pretrained(BERT_PRETAIN_PATH)
		self.vocab = load_data(chinese_vocab)
		self.train_x = None
		self.train_y = None
		self.test_x = None
		self.test_y = None
		self.tag2id = None
		if pre_process:
			self.pre_process()
		else:
			self.load_data()

	def padding(self, x, y):
		x_data = []
		y_data = []
		masks = []
		max_len = max([len(kk)+2 for kk in x])
		segment_ids = [[0] * max_len for _ in range(len(y))]
		for seq, label in zip(x, y):
			for ii, word in enumerate(seq):
				if word not in self.vocab:
					seq[ii] = '[UNK]'
			seq = seq[:max_len - 2]
			label = label[:max_len - 2]
			seq = ['[CLS]'] + seq + ['[SEP]']
			label = ['[CLS]'] + label + ['[SEP]']
			seq_len = len(seq)
			pad = [0] * (max_len - seq_len)
			seq_id = self.tokenizer.convert_tokens_to_ids(seq) + pad
			label_id = [self.tag2id.get(tag, 0) for tag in label] + pad
			x_data.append(seq_id)
			y_data.append(label_id)
			mask = [1] * seq_len + pad
			masks.append(mask)
		x_data, y_data, segment_ids = th.Tensor(x_data).long(), th.Tensor(y_data).long(), th.Tensor(segment_ids).long()
		masks = th.Tensor(masks).long()
		return x_data, y_data, masks, segment_ids

	def generate_data(self, data_type, batch_size):
		if data_type == 'train':
			x, y = self.train_x, self.train_y
		else:
			x, y = self.test_x, self.test_y
		sample_size = len(y)
		total_epoch = sample_size // batch_size
		print("total epoch:\t{}\tbatch_size:\t{}".format(total_epoch, batch_size))
		data = []
		for ii in range(total_epoch):
			start, end = ii * batch_size, (ii + 1) * batch_size
			if start >= sample_size:
				continue
			if end >= sample_size:
				end = sample_size
			x_data, y_data = x[start:end], y[start:end]
			x_data, y_data, masks, segment_ids = self.padding(x_data, y_data)
			data.append([x_data, y_data, masks, segment_ids])
		return data

	def load_data(self):
		data = load_json_data(BERT_DATA_JSON)
		self.train_x = data['train_x']
		self.train_y = data['train_y']
		self.test_x = data['test_x']
		self.test_y = data['test_y']
		self.tag2id = load_json_data(BERT_TAG2ID_JSON)

	def pre_process(self):
		self.train_x, self.train_y = read_label_data(TRAIN_PATH)
		self.test_x, self.test_y = read_label_data(TEST_PATH)
		labels = set()
		for label in self.train_y:
			labels.update(label)
		labels.update({'[CLS]', '[SEP]'})
		labels = list(labels)
		self.tag2id = {label: idx for idx, label in enumerate(labels)}
		data = {
			"train_x": self.train_x, "train_y": self.train_y,
			"test_x": self.test_x, "test_y": self.test_y,
		}
		save_json_data(data, BERT_DATA_JSON)
		save_json_data(self.tag2id, BERT_TAG2ID_JSON)


class TokenizerElmoText(object):
	def __init__(self, use_cuda, pre_process=False):
		self.simplified_to_traditional = opencc.OpenCC('s2t')
		self.use_cuda = use_cuda
		self.train_x = None
		self.train_y = None
		self.test_x = None
		self.test_y = None
		self.tag2id = None
		if pre_process:
			self.pre_process()
		else:
			self.load_data()

	def padding(self, x, y):
		x_data = []
		y_data = []
		masks = []
		max_len = max([len(kk) for kk in x])
		for seq, label in zip(x, y):
			seq_len = len(seq)
			seq = seq[:max_len] + [PAD] * (max_len - seq_len)
			label = label[:max_len]
			pad = [0] * (max_len - seq_len)
			label_id = [self.tag2id.get(tag, 0) for tag in label] + pad
			x_data.append(seq)
			y_data.append(label_id)
			mask = [1] * seq_len + pad
			masks.append(mask)
		y_data = th.Tensor(y_data).long()
		masks = th.Tensor(masks).long()
		return x_data, y_data, masks

	def generate_data(self, data_type, batch_size):
		if data_type == 'train':
			x, y = self.train_x, self.train_y
		else:
			x, y = self.test_x, self.test_y
		sample_size = len(y)
		total_epoch = sample_size // batch_size
		print("total epoch:\t{}\tbatch_size:\t{}".format(total_epoch, batch_size))
		data = []
		for ii in range(total_epoch):
			start, end = ii * batch_size, (ii + 1) * batch_size
			if start >= sample_size:
				continue
			if end >= sample_size:
				end = sample_size
			x_data, y_data = x[start:end], y[start:end]
			x_data, y_data, masks = self.padding(x_data, y_data)
			data.append([x_data, y_data, masks])
		return data

	def load_data(self):
		data = load_json_data(ELMO_DATA_JSON)
		self.train_x = data['train_x']
		self.train_y = data['train_y']
		self.test_x = data['test_x']
		self.test_y = data['test_y']
		self.tag2id = load_json_data(ELMO_TAG2ID_JSON)

	def pre_process(self):
		self.train_x, self.train_y = read_label_data(TRAIN_PATH)
		self.test_x, self.test_y = read_label_data(TEST_PATH)
		labels = set()
		for label in self.train_y:
			labels.update(label)
		self.train_x = self.s2t(self.train_x)
		self.test_x = self.s2t(self.test_x)
		labels = list(labels)
		self.tag2id = {label: idx for idx, label in enumerate(labels)}
		data = {
			"train_x": self.train_x, "train_y": self.train_y,
			"test_x": self.test_x, "test_y": self.test_y,
		}
		save_json_data(data, ELMO_DATA_JSON)
		save_json_data(self.tag2id, ELMO_TAG2ID_JSON)

	def s2t(self, data):
		text = []
		for line in data:
			s = self.simplified_to_traditional.convert(' '.join(line))
			text.append(s.split(' '))
		return text


def generate_batch_data(x, y, batch_size, use_cuda=False):
	"""
	生成batch 数据
	:param x:
	:param y:
	:param batch_size:
	:param use_cuda:
	:return:
	"""
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
		x_data, seq_lens = padding_sentence(x_data)
		y_data, _ = padding_sentence(y_data)
		if use_cuda:
			x_data, y_data = x_data.cuda(), y_data.cuda()
		x_data, y_data = Variable(x_data).long(), Variable(y_data).long()
		yield x_data, y_data, seq_lens


def padding_sentence(data, default=0):
	seq_lens = [len(kk) for kk in data]
	max_len = max(seq_lens)
	for kk, sentence in enumerate(data):
		seq_len = len(sentence)
		if seq_len >= max_len:
			sentence = sentence[:max_len]
		else:
			sentence += [default] * (max_len - seq_len)
		data[kk] = sentence
	data = th.Tensor(data)
	return data, seq_lens


def mask_targets(targets, sequence_lengths, batch_first=False):
	""" Masks the targets """
	if not batch_first:
		targets = targets.transpose(0, 1)
	t = []
	for l, p in zip(targets, sequence_lengths):
		t.append(l[:p].data.tolist())
	return t


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
