#!/usr/bin python3
# -*- coding: utf-8 -*-
# @时间   : 19-4-4 下午6:15
# @作者   : Lin lifang
# @文件   : lstm_crf.py
import torch.nn as nn
import torch.nn.functional as F
from crf import CRF


class LSTMCRF(nn.Module):
	def __init__(self, vocab_size, embed_size, num_units, num_tag):
		super(LSTMCRF, self).__init__()
		self.num_tag = num_tag
		self.crf = CRF(num_tag)
		self.embedding = nn.Embedding(vocab_size, embed_size)
		self.gru = nn.GRU(embed_size, num_units, batch_first=True, bidirectional=True)
		self.linear = nn.Linear(num_units, num_tag)

	def forward(self, x):
		mask, _ = x.ge(0).float().sum(dim=1)
		out = self.rnn_layer(x)
		loss = self.crf(out, mask)
		return -loss

	def test(self, x):
		mask, _ = x.ge(0).float().sum(dim=1)
		out = self.rnn_layer(x)
		loss, max_score, max_score_pre = self.crf(out)
		best_paths = self.crf.viterbi(max_score, max_score_pre, mask)
		return best_paths

	def rnn_layer(self, x):
		embed = self.embedding(x)
		out, _ = self.gru(embed)
		out = F.softmax(self.linear(out), dim=-1)
		return out
