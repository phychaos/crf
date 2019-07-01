#!/usr/bin python3
# -*- coding: utf-8 -*-
import os

from torch import optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from core.metric import get_ner_fmeasure, recover_label, recover_bert_label
from config.config import VOCAB_DATA, TAG2ID_JSON, EMBEDDING_DATA, epochs, MODEL_PATH
from config.config import HyperParams as hp
from core.preprocess import load_json_data, load_train_data, generate_batch_data, preprocess, TokenizerElmoText
from model.rnn_crf import RNNCRF
import torch as th

use_cuda = th.cuda.is_available()


def run_rnn():
	# preprocess() # 预处理
	# exit()
	train_x, train_y, test_x, test_y = load_train_data()
	vocab = load_json_data(VOCAB_DATA)
	tag2id = load_json_data(TAG2ID_JSON)
	id2tag = {str(idx): tag for tag, idx in tag2id.items()}
	batch_size = hp.batch_size
	
	model = RNNCRF(len(vocab), hp.embed_size, hp.num_units, hp.num_layers, len(tag2id), None, use_cuda)
	if use_cuda:
		model.cuda()
	optimizer = optim.Adam(model.parameters())
	scheduler = StepLR(optimizer, step_size=1, gamma=hp.lr_decay)
	
	def fit():
		_train_loss = 0
		step = 0
		model.train()
		for x_data, y_data, seq_lens in tqdm(generate_batch_data(train_x, train_y, batch_size, use_cuda), desc='训练'):
			step += 1
			optimizer.zero_grad()
			loss = model(x_data, y_data, seq_lens)
			_train_loss += float(loss.item())
			loss.backward()
			optimizer.step()
		return _train_loss / step
	
	def test():
		_test_loss = 0
		paths = []
		step = 0
		model.eval()
		with th.no_grad():
			for x_data, y_data, seq_lens in tqdm(generate_batch_data(test_x, test_y, batch_size, use_cuda), desc='测试'):
				step += 1
				loss, _best_paths = model.test(x_data, y_data, seq_lens)
				paths.extend(_best_paths)
				_test_loss += float(loss.item())
		return _test_loss / step, paths
	
	source_tag = recover_label(test_y, id2tag)
	for epoch in range(epochs):
		scheduler.step()
		train_loss = fit()
		test_loss, best_paths = test()
		predicts = recover_label(best_paths, id2tag)
		acc, p, r, f = get_ner_fmeasure(source_tag, predicts)
		print('epoch:\t{}\ttrain loss:\t{}\tdev loss:\t{}'.format(epoch, round(train_loss, 4), round(test_loss, 4)))
		print('acc:\t{}\tp:\t{}\tr:\t{}\tf:\t{}'.format(acc, p, r, f))
		print('****************************************************\n\n')


def run_bert():
	from config.config import BertParams as bp
	from core.preprocess import TokenizerBertText
	from model.bert_crf import BertNer
	from pytorch_pretrained_bert.optimization import BertAdam
	device = th.device('cuda' if use_cuda else 'cpu')
	token_text = TokenizerBertText(use_cuda, pre_process=False)
	tag2id = token_text.tag2id
	id2tag = {str(idx): tag for tag, idx in tag2id.items()}
	num_tags = len(tag2id)
	model = BertNer(bp.num_units, bp.rnn_hidden, num_tags, bp.num_layers)
	model.to(device)
	optimizer = BertAdam(model.parameters(), lr=bp.lr, warmup=0.1)
	
	def fit():
		_train_loss = 0
		ii = 0
		for ii, batch in enumerate(tqdm(token_text.generate_data('train', bp.batch_size), desc='训练')):
			batch = (t.to(device) for t in batch)
			x_data, y_data, masks, segment_ids = batch
			optimizer.zero_grad()
			loss = model(x_data, y_data, masks, segment_ids)
			_train_loss += float(loss.item())
			loss.backward()
			optimizer.step()
		return round(_train_loss / ii, 4)
	
	def test():
		_test_loss = 0
		paths = []
		ii = 0
		with th.no_grad():
			for ii, batch in enumerate(tqdm(token_text.generate_data('test', bp.batch_size), '测试')):
				batch = (t.to(device) for t in batch)
				x_data, y_data, masks, segment_ids = batch
				loss = model(x_data, y_data, masks, segment_ids)
				_test_loss += float(loss.item())
				_best_paths = model.test(x_data, masks, segment_ids)
				paths.extend(_best_paths)
		return round(_test_loss // ii, 4), paths
	
	source_tag = token_text.test_y
	for epoch in range(epochs):
		train_loss = fit()
		test_loss, best_paths = test()
		predicts = recover_bert_label(best_paths, id2tag)
		acc, p, r, f = get_ner_fmeasure(source_tag, predicts)
		print('epoch:\t{}\ttrain loss:\t{}\tdev loss:\t{}'.format(epoch, round(train_loss, 2), round(test_loss, 2)))
		print('acc:\t{}\tp:\t{}\tr:\t{}\tf:\t{}'.format(acc, p, r, f))
		print('****************************************************\n\n')


def run_elmo():
	from model.elmo_crf import test, ElmoNer
	from config.config import ElmoParams as ep
	from torch.optim import Adam
	device = th.device('cuda' if use_cuda else 'cpu')
	token_text = TokenizerElmoText(use_cuda, pre_process=False)
	tag2id = token_text.tag2id
	id2tag = {str(idx): tag for tag, idx in tag2id.items()}
	num_tags = len(tag2id)
	model = ElmoNer(ep.num_units, ep.rnn_hidden, num_tags, ep.num_layers, use_cuda)
	model.to(device)
	optimizer = Adam(model.parameters(), lr=ep.lr)
	
	def fit():
		_train_loss = 0
		ii = 0
		for ii, batch in enumerate(tqdm(token_text.generate_data('train', ep.batch_size), desc='训练')):
			x_data, y_data, masks = batch
			masks, y_data = masks.to(device), y_data.to(device)
			optimizer.zero_grad()
			loss = model(x_data, y_data, masks)
			_train_loss += float(loss.item())
			loss.backward()
			optimizer.step()
		return round(_train_loss / ii, 4)
	
	def test():
		_test_loss = 0
		paths = []
		ii = 0
		with th.no_grad():
			for ii, batch in enumerate(tqdm(token_text.generate_data('test', ep.batch_size), '测试')):
				x_data, y_data, masks = batch
				masks, y_data = masks.to(device), y_data.to(device)
				
				loss = model(x_data, y_data, masks)
				_test_loss += float(loss.item())
				_best_paths = model.test(x_data, masks)
				paths.extend(_best_paths)
		return round(_test_loss // ii, 4), paths
	
	source_tag = token_text.test_y
	for epoch in range(epochs):
		train_loss = fit()
		test_loss, best_paths = test()
		predicts = recover_label(best_paths, id2tag)
		acc, p, r, f = get_ner_fmeasure(source_tag, predicts)
		print('epoch:\t{}\ttrain loss:\t{}\tdev loss:\t{}'.format(epoch, round(train_loss, 2), round(test_loss, 2)))
		print('acc:\t{}\tp:\t{}\tr:\t{}\tf:\t{}'.format(acc, p, r, f))
		print('****************************************************\n\n')


if __name__ == '__main__':
	run_rnn()
# run_bert()
# run_elmo()
