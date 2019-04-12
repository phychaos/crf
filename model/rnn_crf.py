#!/usr/bin python3
# -*- coding: utf-8 -*-
# @时间   : 19-4-4 下午6:15
# @作者   : Lin lifang
# @文件   : rnn_crf.py
import torch.nn as nn
from model.crf import CRF
from torch.autograd import Variable
import torch as th


class RNNCRF(nn.Module):
	def __init__(self, vocab_size, embed_size, num_units, num_layers, num_tag, pre_train, use_cuda):
		super(RNNCRF, self).__init__()
		self.num_tag = num_tag
		self.use_cuda = use_cuda
		self.crf = CRF(num_tag)
		self.embedding = nn.Embedding(vocab_size, embed_size, _weight=pre_train)
		self.rnn = nn.LSTM(embed_size, num_units, num_layers=num_layers, batch_first=True, bidirectional=True)
		self.linear = nn.Linear(2 * num_units, num_tag)

	def forward(self, x, y, seq_lens):
		"""
		训练模型
		:param x:
		:param y: batch * seq_len
		:param seq_lens:
		:return:
		"""
		emissions, mask = self.rnn_layer(x, seq_lens)
		loss = -1 * self.crf(emissions, y.transpose(0, 1), mask)
		return loss

	def test(self, x, y, seq_lens):
		emissions, mask = self.rnn_layer(x, seq_lens)
		loss = -1 * self.crf(emissions, y.transpose(0, 1), mask)
		best_paths = self.crf.decode(emissions, mask=mask)
		return loss, best_paths

	def rnn_layer(self, x, seq_lens):
		"""
		输出发射概率
		:param x:
		:param seq_lens:
		:return: seq_len batch_size 2*num_units
		"""
		batch_size, max_len = x.size()
		mask = create_mask(seq_lens, batch_size, max_len, self.use_cuda)
		embed = self.embedding(x)
		out, _ = self.rnn(embed)
		out = self.linear(out)
		out = out.transpose(0, 1)
		mask = mask.transpose(0, 1)
		return out, mask


def create_mask(seq_lens, batch_size, max_len, cuda, batch_first=True):
	""" Creates binary mask """
	mask = Variable(th.ones(batch_size, max_len).type(th.ByteTensor))
	if cuda:
		mask = mask.cuda()
	for i, l in enumerate(seq_lens):
		if batch_first:
			if l < max_len:
				mask.data[i, l:] = 0
		else:
			if l < max_len:
				mask.data[l:, i] = 0
	return mask
