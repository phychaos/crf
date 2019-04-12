#!/usr/bin python3
# -*- coding: utf-8 -*-
# @时间   : 19-4-8 上午9:41
# @作者   : Lin lifang
# @文件   : bert_crf.py
from pytorch_pretrained_bert import BertModel, BertTokenizer
from config.config import BERT_PRETAIN_PATH
from model.crf import CRF
import torch
import torch.nn as nn


class BertNer(nn.Module):
	def __init__(self, num_units, rnn_hidden, num_tags, num_layers=1):
		super(BertNer, self).__init__()
		self.bert_model = BertModel.from_pretrained(BERT_PRETAIN_PATH)
		self.rnn = nn.GRU(num_units, rnn_hidden, num_layers=num_layers, batch_first=True, bidirectional=True)
		self.linear = nn.Linear(2 * rnn_hidden, num_tags)
		# self.linear = nn.Linear(num_units, num_tags)
		self.crf = CRF(num_tags)

	def forward(self, x_data, y_data, masks, segment_ids):
		"""
		前向算法
		:param x_data:
		:param y_data:
		:param masks:
		:param segment_ids:
		:return:
		"""
		encoded_layers, _ = self.bert_model(x_data, segment_ids, masks, False)
		out = self.rnn_layer(encoded_layers)
		loss = -1 * self.crf(out, y_data.transpose(0, 1), masks.transpose(0, 1))
		return loss

	def rnn_layer(self, encoded_layers):
		out, _ = self.rnn(encoded_layers)
		out = self.linear(out)
		out = out.transpose(0, 1)
		return out

	def test(self, x_data, masks, segment_ids):
		encoded_layers, _ = self.bert_model(x_data, segment_ids, masks, False)

		out = self.rnn_layer(encoded_layers)
		best_paths = self.crf.decode(out, mask=masks.transpose(0, 1))
		return best_paths


def test():
	tokenizer = BertTokenizer.from_pretrained(BERT_PRETAIN_PATH)
	model = BertModel.from_pretrained(BERT_PRETAIN_PATH)
	text = "[CLS]你好,吃早饭了吗？[SEP]"

	tokenized_text = tokenizer.tokenize(text)

	token_id = tokenizer.convert_tokens_to_ids(tokenized_text)
	segments_ids = [0] * len(token_id)
	tokens_tensor = torch.tensor([token_id])
	segments_tensors = torch.tensor([segments_ids])

	with torch.no_grad():
		encoded_layers, _ = model(tokens_tensor, segments_tensors)

	print(len(encoded_layers))
	for ii, encoded_layer in enumerate(encoded_layers):
		print(ii, encoded_layer.size())
