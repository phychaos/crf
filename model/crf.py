#!/usr/bin python3
# -*- coding: utf-8 -*-
# @时间   : 19-4-4 下午3:04
# @作者   : Lin lifang
# @文件   : crf.py
import copy

import torch as th
from typing import List, Optional, Union
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class NeuralCRF(nn.Module):
	def __init__(self, num_tag, embed_size=100, hidden_size=100, use_cuda=False):
		super(NeuralCRF, self).__init__()
		self.num_tag = num_tag
		self.y0 = num_tag
		self.topk = num_tag
		self.hidden_size = hidden_size
		self.use_cuda = use_cuda
		self.device = th.device('cuda' if use_cuda else 'cpu')
		self.embedding = nn.Embedding(num_tag + 1, embed_size)
		self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
		self.linear = nn.Linear(hidden_size, num_tag)
	
	def init_hidden(self, batch_size):
		h = Variable(th.zeros(1, batch_size, self.hidden_size)).to(self.device)
		return h
	
	def forward(self, emissions, tags, mask):
		"""
		crf 极大似然估计计算损失函数 w*f - logz
		:param emissions: 发射概率
		:param tags: 标签
		:param mask: pad
		:return: log(p)
		"""
		mask = mask.float()
		seq_len, batch_size, num_tag = emissions.size()
		path_score = self.path_score(emissions, tags, batch_size, seq_len, mask)
		scores = self.log_partition(emissions, batch_size, seq_len, mask)
		
		log_z = self._log_sum_exp(scores,  dim=1)
		likelihood = path_score - log_z
		return likelihood.sum()
	
	def rnn_forward(self, batch_size, x=None, h=None):
		if x is None:
			x = Variable(th.ones((batch_size, 1), ).long() * self.y0).to(self.device)
		
		if h is None:
			h = self.init_hidden(batch_size)
		x = self.embedding(x)
		out, h = self.gru(x, h)
		return out, h
	
	def path_score(self, emissions, tags, batch_size, seq_len, mask):
		"""
		路径分数
		:param emissions:
		:param tags:
		:param batch_size:
		:param seq_len:
		:param mask:
		:return: w*f 真是路径特征分数
		"""
		out, h = self.rnn_forward(batch_size)
		transitions = self.linear(out.squeeze(1))
		cur_tag = tags[0].view(-1, 1)
		scores = transitions.gather(1, cur_tag).squeeze(1) + emissions[0].gather(1, cur_tag).squeeze(1)
		scores = scores * mask[0]
		
		for i in range(1, seq_len):
			pre_tag, cur_tag = cur_tag, tags[i].view(-1, 1)
			yi = self.embedding(pre_tag)
			out, h = self.gru(yi, h)
			transitions = self.linear(out.squeeze(1))
			current_score = transitions.gather(1, cur_tag).squeeze(1) + emissions[0].gather(1, cur_tag).squeeze(1)
			scores += current_score * mask[i]
		return scores
	
	def beam_search(self, x, emission, scores, mask, batch_size):
		"""
		束搜索最佳路径
		:param x:
		:param emission: 发射概率 batch * num_tag
		:param scores: t-1时刻得分 batch * num_tag
		:param mask: mask batch
		:param batch_size: 数据量大小
		:return:
		"""
		path = [[] for _ in range(batch_size)]
		_score = None
		_index = None
		for pre_tag, pre_h, _path in x:
			out, h = self.rnn_forward(batch_size, pre_tag, pre_h)  # batch * 1 * hidden and 1 * batch * hidden
			transitions = self.linear(out.squeeze(1))  # batch * num_tag
			# t-1 score + transitions + emission batch * num_tag
			score = scores.gather(1, pre_tag) + (transitions + emission) * mask.unsqueeze(1)
			
			# score 合并 batch * (num_tag*num_tag)
			
			if _score is None:
				_score = score
			else:
				_score = th.cat((_score, score), dim=1)
			
			for bt in range(batch_size):
				cur_path = _path[bt]
				for kk in range(self.topk):
					path[bt].append([cur_path + [kk], kk, h[:, bt, :].view(1, 1, -1)])
		# 选取前k个值 batch * num_tag
		_scores, index = _score.topk(self.topk)
		# 排序后前一刻节点和隐藏层， 历史路径记录
		inputs = []
		
		for ii in range(self.topk):
			tag = Variable(th.zeros((batch_size, 1))).long().to(self.device)
			cur_path = []
			hidden = None
			for bt in range(batch_size):
				pre_path, cur_tag, h = path[bt][index[bt, ii].item()]
				if hidden is None:
					hidden = h
				else:
					hidden = th.cat((hidden, h), dim=1)
				cur_path.append(pre_path)
				tag[bt, 0] = cur_tag
			inputs.append([tag, hidden, cur_path])
		
		return inputs, _scores
	
	def beam_search_decode(self, emissions, batch_size, seq_len, mask=None):
		out, h = self.rnn_forward(batch_size)
		transitions = self.linear(out.squeeze(1))
		scores = transitions + emissions[0]
		scores = scores * mask[0].unsqueeze(1)
		# 选取前k个值 batch * num_tag
		beam = Beam(scores, h, self.topk, seq_len)
		nodes = beam.get_next_nodes()
		for t in range(1, seq_len):
			siblings = []
			for inputs, hidden in nodes:
				inputs = inputs.long().to(self.device)
				out, h = self.rnn_forward(batch_size, inputs, hidden)
				transitions = self.linear(out.squeeze(1))
				score = transitions + emissions[t]
				score = score * mask[t].unsqueeze(1)
				siblings.append([score, h])
			nodes = beam.select_k(siblings, t)
	
	def log_partition(self, emissions, batch_size, seq_len, mask=None):
		"""
		beam search 近似计算logz
		:param emissions:
		:param batch_size:
		:param seq_len:序列长度
		:param mask:
		:return:
		"""
		out, h = self.rnn_forward(batch_size)
		transitions = self.linear(out.squeeze(1))
		scores = transitions + emissions[0]
		scores = scores * mask[0].unsqueeze(1)
		# 选取前k个值 batch * num_tag
		scores, index = scores.topk(self.topk)
		# 排序后前一刻节点和隐藏层， 历史路径记录
		inputs = []
		for ii in range(self.topk):
			tag = Variable(index[:, ii].unsqueeze(1)).long().to(self.device)
			cur_path = index[:, ii].unsqueeze(1).cpu().tolist()
			inputs.append([tag, h, cur_path])
		
		for i in range(1, seq_len):
			inputs, scores = self.beam_search(inputs, emissions[i], scores, mask[i], batch_size)
		# log_prob = self._log_sum_exp(scores, dim=1)
		return scores
	
	@staticmethod
	def _log_sum_exp(score, dim):
		"""
		A + log sum(exp(x-A)) 防止溢出
		:param score: batch * num_tag
		:param dim: dim=1
		:return: log_prob
		"""
		max_score, _ = score.max(dim=dim)
		log_prob = th.log(th.exp(score - max_score.unsqueeze(dim)).sum(dim=dim))
		return max_score + log_prob
	
	def decode(self, emissions, mask=None):
		seq_len, batch_size, _ = emissions.size()
		if mask is None:
			th.ones(seq_len, batch_size)
		mask = mask.float()
		return self._viterbi(emissions, batch_size, seq_len, mask)
	
	def search_path(self, batch_size, scores, x, mask):
		"""
		最佳路径
		:param batch_size:
		:param scores:
		:param x:
		:param mask:
		:return:
		"""
		seq_length = mask.long().sum(0)
		all_path = []
		for bt in range(batch_size):
			_, _, path = x[0]
			cur_path = path[bt][:seq_length[bt]]
			all_path.append(cur_path)
		return all_path
	
	def _viterbi(self, emissions, batch_size, seq_len, mask):
		"""
		维特比反向算法
		:param emissions: seq_len * batch * num_tag
		:param mask:
		:return:
		"""
		
		out, h = self.rnn_forward(batch_size)
		transitions = self.linear(out.squeeze(1))
		scores = transitions + emissions[0]
		scores = scores * mask[0].unsqueeze(1)
		x = [[Variable(th.ones((batch_size, 1), ).long() * kk).to(self.device), h, [[kk] for _ in range(batch_size)]]
			 for kk in range(self.num_tag)]
		for i in range(1, seq_len):
			x, scores = self.beam_search(x, emissions[i], scores, mask[i], batch_size)
		# log_prob = self._log_sum_exp(scores, dim=1)
		all_path = self.search_path(batch_size, scores, x, mask)
		return all_path


class Beam:
	def __init__(self, score, hidden, num_beam, seq_len):
		"""
		root : (score, hidden)
		batch * vocab_size
		"""
		# score, hidden = root
		score = score.cpu()
		self.num_beam = num_beam
		self.seq_len = seq_len
		self.batch_size = score.size()[0]
		self.hidden = th.zeros_like(hidden)
		s, i = score.topk(num_beam)
		# s = s.data
		i = i.data
		self.beams = []
		for ii in range(num_beam):
			path = th.zeros(self.batch_size, seq_len)
			path[:, 0] = i[:, ii]
			beam = [s[:, ii], path, hidden]
			self.beams.append(beam)
	
	def select_k(self, siblings, t):
		"""
		siblings : [score,hidden]
		"""
		candidate = []
		for p_index, (score, hidden) in enumerate(siblings):
			score = score.cpu()
			parents = self.beams[p_index]  # (cummulated score, list of sequence)
			s, i = score.topk(self.num_beam)
			i = i.data
			batch_size, num_tag = score.size()
			for kk in range(self.num_beam):
				vocab_id = copy.deepcopy(parents[1])
				vocab_id[:, t] = i[:, kk]
				current_score = parents[0] + s[:, kk]
				candidate.append([current_score, vocab_id, hidden])
		# 候选集排序
		beams = [[th.zeros(self.batch_size), th.zeros(self.batch_size, self.seq_len, dtype=th.int),
				  th.zeros_like(self.hidden)] for _ in range(self.num_beam)]
		for ii in range(self.batch_size):
			beam = [[cand[0][ii], cand[1][ii, :], cand[2][:, ii]] for cand in candidate]
			beam = sorted(beam, key=lambda x: x[0], reverse=True)[:self.num_beam]
			for kk in range(self.num_beam):
				beams[kk][0][ii] = beam[kk][0]
				beams[kk][1][ii, :] = beam[kk][1]
				beams[kk][2][:, ii] = beam[kk][2]
		self.beams = beams
		# last_input, hidden
		return [[b[1][:, t], b[2]] for b in self.beams]
	
	def get_best_seq(self):
		return self.beams[0][1]
	
	def get_next_nodes(self):
		return [[b[1][:, 0], b[2]] for b in self.beams]


class CRF1(nn.Module):
	def __init__(self, num_tag):
		super(CRF1, self).__init__()
		self.num_tag = num_tag
		self.start_transitions = nn.Parameter(th.Tensor(num_tag))
		self.transitions = nn.Parameter(th.Tensor(num_tag, num_tag))
		self.end_transitions = nn.Parameter(th.Tensor(num_tag))
		self.reset_params()
	
	def reset_params(self):
		nn.init.uniform_(self.start_transitions, -0.1, 0.1)
		nn.init.uniform_(self.transitions, -0.1, 0.1)
		nn.init.uniform_(self.end_transitions, -0.1, 0.1)
	
	def forward(self, emissions, tags, mask=None):
		"""
		crf 极大似然估计计算损失函数 w*f - logz
		:param emissions: 发射概率
		:param tags: 标签
		:param mask: pad
		:return: log(p)
		"""
		batch_size, seq_len, num_tag = emissions.size()
		path_score = self.path_score(emissions, tags, seq_len, mask)
		log_z = self.log_partition(emissions, seq_len, mask)
		likelihood = path_score - log_z
		return likelihood.sum()
	
	def path_score(self, emissions, tags, seq_len, mask):
		"""
		路径分数
		:param emissions:
		:param tags:
		:param seq_len:
		:param mask:
		:return: w*f 真是路径特征分数
		"""
		mask = mask.float()
		scores = self.start_transitions[tags[0]]
		scores += emissions[0].gather(1, tags[0].view(-1, 1)).squeeze(1) * mask[0]
		for i in range(1, seq_len):
			pre_tag, cur_tag = tags[i - 1], tags[i]
			scores += emissions[i].gather(1, cur_tag.view(-1, 1)).squeeze(1) * mask[i]
			transition_score = self.transitions[pre_tag, cur_tag]
			scores += transition_score * mask[i]
		last_tag_indices = mask.long().sum(0) - 1
		last_tag = tags.gather(0, last_tag_indices.view(1, -1)).squeeze(0)
		scores += self.end_transitions[last_tag]
		return scores
	
	def log_partition(self, emissions, seq_len, mask=None):
		"""
		前向算法 alpha  = dot(alpha, M) = exp(log alpha + log M) 计算logz
		:param emissions:
		:param seq_len:
		:param mask:
		:return:
		"""
		log_prob = self.start_transitions.view(1, -1) + emissions[0]
		mask = mask.float()
		for ii in range(1, seq_len):
			emissions_score = emissions[ii].unsqueeze(1)  # batch * 1 * num_tag
			transitions_score = self.transitions.unsqueeze(0)  # 1 * num_tag * num_tag
			score = log_prob.unsqueeze(2) + emissions_score + transitions_score  # alpha_T + t + s
			score = self._log_sum_exp(score, dim=1)
			log_prob = score * mask[ii].unsqueeze(1) + log_prob * (1. - mask[ii]).unsqueeze(1)  # batch * num_tag
		log_prob += self.end_transitions.view(1, -1)
		return self._log_sum_exp(log_prob, dim=1)  # batch
	
	@staticmethod
	def _log_sum_exp(score, dim):
		"""
		A + log sum(exp(x-A)) 防止溢出
		:param score: batch * num_tag * num_tag
		:param dim: dim=1
		:return: log_prob
		"""
		max_score, _ = score.max(dim=dim)
		log_prob = th.log(th.exp(score - max_score.unsqueeze(dim)).sum(dim=dim))
		return max_score + log_prob
	
	def decode(self, emissions, mask=None):
		seq_len, batch_size, _ = emissions.size()
		if mask is None:
			th.ones(seq_len, batch_size)
		return self._viterbi(emissions, batch_size, seq_len, mask)
	
	def _viterbi(self, emissions, batch_size, seq_len, mask):
		"""
		维特比反向算法
		:param emissions: seq_len * batch * num_tag
		:param mask:
		:return:
		"""
		current_score = self.start_transitions.view(1, -1) + emissions[0]
		viterbi_scores = []
		viterbi_path = []
		viterbi_scores.append(current_score)
		for ii in range(1, seq_len):
			emissions_score = emissions[ii].unsqueeze(1)  # batch * 1 * num_tag
			transitions_score = self.transitions.unsqueeze(0)  # batch * num_tag * num_tag
			score = viterbi_scores[ii - 1].unsqueeze(2) + emissions_score + transitions_score
			best_score, best_path = score.max(dim=1)
			viterbi_path.append(best_path)
			viterbi_scores.append(best_score)
		seq_length = mask.long().sum(0) - 1
		all_path = []
		for idx in range(batch_size):
			end = seq_length[idx]
			last_score = viterbi_scores[end][idx] + self.end_transitions
			_, best_tag = last_score.max(0)
			best_tags = [best_tag.item()]
			for path in reversed(viterbi_path[:end]):
				best_tag = path[idx][best_tag].item()
				best_tags.append(best_tag)
			all_path.append(reversed(best_tags))
		return all_path


class CRF(nn.Module):
	"""Conditional random field.
	This module implements a conditional random field [LMP]. The forward computation
	of this class computes the log likelihood of the given sequence of tags and
	emission score tensor. This class also has ``decode`` method which finds the
	best tag sequence given an emission score tensor using `Viterbi algorithm`_.
	Arguments
	---------
	num_tags : int
		Number of tags.
	Attributes
	----------
	num_tags : int
		Number of tags passed to ``__init__``.
	start_transitions : :class:`~torch.nn.Parameter`
		Start transition score tensor of size ``(num_tags,)``.
	end_transitions : :class:`~torch.nn.Parameter`
		End transition score tensor of size ``(num_tags,)``.
	transitions : :class:`~torch.nn.Parameter`
		Transition score tensor of size ``(num_tags, num_tags)``.
	References
	----------
	.. [LMP] Lafferty, J., McCallum, A., Pereira, F. (2001).
			 "Conditional random fields: Probabilistic models for segmenting and
			 labeling sequence data". *Proc. 18th International Conf. on Machine
			 Learning*. Morgan Kaufmann. pp. 282–289.
	.. _Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm
	"""
	
	def __init__(self, num_tags):
		if num_tags <= 0:
			raise ValueError('invalid number of tags: {}'.format(num_tags))
		super().__init__()
		self.num_tags = num_tags
		self.start_transitions = nn.Parameter(th.Tensor(num_tags))
		self.end_transitions = nn.Parameter(th.Tensor(num_tags))
		self.transitions = nn.Parameter(th.Tensor(num_tags, num_tags))
		self.reset_parameters()
	
	def reset_parameters(self):
		"""Initialize the transition parameters.
		The parameters will be initialized randomly from a uniform distribution
		between -0.1 and 0.1.
		"""
		nn.init.uniform_(self.start_transitions, -0.1, 0.1)
		nn.init.uniform_(self.end_transitions, -0.1, 0.1)
		nn.init.uniform_(self.transitions, -0.1, 0.1)
	
	def __repr__(self) -> str:
		return '{}(num_tags={})'.format(self.__class__.__name__, self.num_tags)
	
	def forward(self, emissions, tags, mask=None, reduce=True):
		"""Compute the log likelihood of the given sequence of tags and emission score.
		Arguments
		---------
		emissions : :class:`~torch.autograd.Variable`
			Emission score tensor of size ``(seq_length, batch_size, num_tags)``.
		tags : :class:`~torch.autograd.Variable`
			Sequence of tags as ``LongTensor`` of size ``(seq_length, batch_size)``.
		mask : :class:`~torch.autograd.Variable`, optional
			Mask tensor as ``ByteTensor`` of size ``(seq_length, batch_size)``.
		reduce : bool
			Whether to sum the log likelihood over the batch.
		Returns
		-------
		:class:`~torch.autograd.Variable`
			The log likelihood. This will have size (1,) if ``reduce=True``, ``(batch_size,)``
			otherwise.
		"""
		assert emissions.dim() == 3 and tags.dim() == 2
		assert emissions.size()[:2] == tags.size()
		assert emissions.size(2) == self.num_tags
		if mask is None:
			mask = Variable(self._new(tags.size()).fill_(1)).byte()
		else:
			assert mask.size() == tags.size()
			assert all(mask[0].data)
		
		numerator = self._compute_joint_llh(emissions, tags, mask)
		denominator = self._compute_log_partition_function(emissions, mask)
		llh = numerator - denominator
		return llh if not reduce else th.sum(llh)
	
	def decode(self, emissions, mask=None):
		"""Find the most likely tag sequence using Viterbi algorithm.
		Arguments
		---------
		emissions : :class:`~torch.autograd.Variable` or :class:`~torch.FloatTensor`
			Emission score tensor of size ``(seq_length, batch_size, num_tags)``.
		mask : :class:`~torch.autograd.Variable` or :class:`torch.ByteTensor`
			Mask tensor of size ``(seq_length, batch_size)``.
		Returns
		-------
		list
			List of list containing the best tag sequence for each batch.
		"""
		assert emissions.dim() == 3
		assert emissions.size(2) == self.num_tags
		assert mask.size() == emissions.size()[:2]
		
		if isinstance(emissions, Variable):
			emissions = emissions.data
		if mask is None:
			mask = self._new(emissions.size()[:2]).fill_(1).byte()
		elif isinstance(mask, Variable):
			mask = mask.data
		
		return self._viterbi_decode(emissions, mask)
	
	def _compute_joint_llh(self, emissions, tags, mask):
		# emissions: (seq_length, batch_size, num_tags)
		# tags: (seq_length, batch_size)
		# mask: (seq_length, batch_size)
		assert emissions.dim() == 3 and tags.dim() == 2
		assert emissions.size()[:2] == tags.size()
		assert emissions.size(2) == self.num_tags
		assert mask.size() == tags.size()
		assert all(mask[0].data)
		
		seq_length = emissions.size(0)
		mask = mask.float()
		
		# Start transition score
		llh = self.start_transitions[tags[0]]  # (batch_size,)
		
		for i in range(seq_length - 1):
			cur_tag, next_tag = tags[i], tags[i + 1]
			# Emission score for current tag
			llh += emissions[i].gather(1, cur_tag.view(-1, 1)).squeeze(1) * mask[i]
			# Transition score to next tag
			transition_score = self.transitions[cur_tag, next_tag]
			# Only add transition score if the next tag is not masked (mask == 1)
			llh += transition_score * mask[i + 1]
		
		# Find last tag index
		last_tag_indices = mask.long().sum(0) - 1  # (batch_size,)
		last_tags = tags.gather(0, last_tag_indices.view(1, -1)).squeeze(0)
		
		# End transition score
		llh += self.end_transitions[last_tags]
		# Emission score for the last tag, if mask is valid (mask == 1)
		llh += emissions[-1].gather(1, last_tags.view(-1, 1)).squeeze(1) * mask[-1]
		
		return llh
	
	def _compute_log_partition_function(self, emissions, mask):
		# emissions: (seq_length, batch_size, num_tags)
		# mask: (seq_length, batch_size)
		assert emissions.dim() == 3 and mask.dim() == 2
		assert emissions.size()[:2] == mask.size()
		assert emissions.size(2) == self.num_tags
		assert all(mask[0].data)
		
		seq_length = emissions.size(0)
		mask = mask.float()
		
		# Start transition score and first emission
		log_prob = self.start_transitions.view(1, -1) + emissions[0]
		# Here, log_prob has size (batch_size, num_tags) where for each batch,
		# the j-th column stores the log probability that the current timestep has tag j
		
		for i in range(1, seq_length):
			# Broadcast log_prob over all possible next tags
			broadcast_log_prob = log_prob.unsqueeze(2)  # (batch_size, num_tags, 1)
			# Broadcast transition score over all instances in the batch
			broadcast_transitions = self.transitions.unsqueeze(0)  # (1, num_tags, num_tags)
			# Broadcast emission score over all possible current tags
			broadcast_emissions = emissions[i].unsqueeze(1)  # (batch_size, 1, num_tags)
			# Sum current log probability, transition, and emission scores
			score = broadcast_log_prob + broadcast_transitions \
					+ broadcast_emissions  # (batch_size, num_tags, num_tags)
			# Sum over all possible current tags, but we're in log prob space, so a sum
			# becomes a log-sum-exp
			score = self._log_sum_exp(score, 1)  # (batch_size, num_tags)
			# Set log_prob to the score if this timestep is valid (mask == 1), otherwise
			# leave it alone
			log_prob = score * mask[i].unsqueeze(1) + log_prob * (1. - mask[i]).unsqueeze(1)
		
		# End transition score
		log_prob += self.end_transitions.view(1, -1)
		# Sum (log-sum-exp) over all possible tags
		return self._log_sum_exp(log_prob, 1)  # (batch_size,)
	
	def _viterbi_decode(self, emissions, mask):
		# Get input sizes
		seq_length = emissions.size(0)
		batch_size = emissions.size(1)
		sequence_lengths = mask.long().sum(dim=0)
		
		# emissions: (seq_length, batch_size, num_tags)
		assert emissions.size(2) == self.num_tags
		
		# list to store the decoded paths
		best_tags_list = []
		
		# Start transition
		viterbi_score = []
		viterbi_score.append(self.start_transitions.data + emissions[0])
		viterbi_path = []
		
		# Here, viterbi_score is a list of tensors of shapes of (num_tags,) where value at
		# index i stores the score of the best tag sequence so far that ends with tag i
		# viterbi_path saves where the best tags candidate transitioned from; this is used
		# when we trace back the best tag sequence
		
		# Viterbi algorithm recursive case: we compute the score of the best tag sequence
		# for every possible next tag
		for i in range(1, seq_length):
			# Broadcast viterbi score for every possible next tag
			broadcast_score = viterbi_score[i - 1].view(batch_size, -1, 1)
			# Broadcast emission score for every possible current tag
			broadcast_emission = emissions[i].view(batch_size, 1, -1)
			# Compute the score matrix of shape (batch_size, num_tags, num_tags) where
			# for each sample, each entry at row i and column j stores the score of
			# transitioning from tag i to tag j and emitting
			score = broadcast_score + self.transitions.data + broadcast_emission
			# Find the maximum score over all possible current tag
			best_score, best_path = score.max(1)  # (batch_size,num_tags,)
			# Save the score and the path
			viterbi_score.append(best_score)
			viterbi_path.append(best_path)
		
		# Now, compute the best path for each sample
		for idx in range(batch_size):
			# Find the tag which maximizes the score at the last timestep; this is our best tag
			# for the last timestep
			seq_end = sequence_lengths[idx] - 1
			_, best_last_tag = (viterbi_score[seq_end][idx] + self.end_transitions.data).max(0)
			best_tags = [best_last_tag.item()]  # [best_last_tag[0]] #[best_last_tag.item()]
			
			# We trace back where the best last tag comes from, append that to our best tag
			# sequence, and trace it back again, and so on
			for path in reversed(viterbi_path[:sequence_lengths[idx] - 1]):
				best_last_tag = path[idx][best_tags[-1]]
				best_tags.append(best_last_tag.data.tolist())
			
			# Reverse the order because we start from the last timestep
			best_tags.reverse()
			best_tags_list.append(best_tags)
		return best_tags_list
	
	@staticmethod
	def _log_sum_exp(tensor, dim):
		# Find the max value along `dim`
		offset, _ = tensor.max(dim)
		# Make offset broadcastable
		broadcast_offset = offset.unsqueeze(dim)
		# Perform log-sum-exp safely
		safe_log_sum_exp = th.log(th.sum(th.exp(tensor - broadcast_offset), dim))
		# Add offset back
		return offset + safe_log_sum_exp
	
	def _new(self, *args, **kwargs):
		param = next(self.parameters())
		return param.data.new(*args, **kwargs)
