#!/usr/bin python3
# -*- coding: utf-8 -*-
# @时间   : 19-4-12 下午12:43
# @作者   : Lin lifang
# @文件   : elmo_crf.py
from elmoformanylangs import Embedder

from config.config import ELMO_PRETAIN_PATH
import opencc
from model.crf import CRF
import torch
import torch.nn as nn
import numpy as np


class ElmoNer(nn.Module):
	def __init__(self, num_units, rnn_hidden, num_tags, num_layers=1, use_cuda=False):
		super(ElmoNer, self).__init__()
		self.use_cuda = use_cuda
		self.embedding = Embedder(ELMO_PRETAIN_PATH)
		self.rnn = nn.GRU(num_units, rnn_hidden, num_layers=num_layers, batch_first=True, bidirectional=True)
		self.linear = nn.Linear(2 * rnn_hidden, num_tags)
		# self.linear = nn.Linear(num_units, num_tags)
		self.crf = CRF(num_tags)

	def forward(self, x_data, y_data, masks):
		"""
		前向算法
		:param x_data:
		:param y_data:
		:param masks:
		:return:
		"""
		encoded_layers = self.embedding.sents2elmo(x_data)
		out = self.rnn_layer(encoded_layers)
		loss = -1 * self.crf(out, y_data.transpose(0, 1), masks.transpose(0, 1))
		return loss

	def rnn_layer(self, encoded_layers):
		"""
		batch seq_len hidden
		:param encoded_layers:
		:return: batch seq_len class
		"""
		encoded_layers = np.array(encoded_layers)
		encoded_layers = torch.from_numpy(encoded_layers)
		if self.use_cuda:
			encoded_layers = encoded_layers.cuda()
		out, _ = self.rnn(encoded_layers)
		out = self.linear(out)
		out = out.transpose(0, 1)
		return out

	def test(self, x_data, masks):
		encoded_layers = self.embedding.sents2elmo(x_data)

		out = self.rnn_layer(encoded_layers)
		best_paths = self.crf.decode(out, mask=masks.transpose(0, 1))
		return best_paths


def test():
	import numpy as np
	e = Embedder(ELMO_PRETAIN_PATH)
	text = [['今', '天', '天', '气', '真', '好', '阿'], ['你', '吃', '饭', '了', '吗', '?', 'd']]
	a = e.sents2elmo(text)
	a = np.array(a)
	print(a.shape)
	print(a)

	new_text = [chs_to_cht(line) for line in text]

	b = e.sents2elmo(new_text)
	print(b[0].shape, b[1].shape)
	print(b)


def chs_to_cht(line):
	c = opencc.OpenCC('s2t')
	s = c.convert(' '.join(line))
	return s.split(' ')
