#!/usr/bin python3
# -*- coding: utf-8 -*-
# @时间   : 19-4-4 下午3:04
# @作者   : Lin lifang
# @文件   : crf.py
import copy

import torch as th
from torch.autograd import Variable
import torch.nn as nn


class NeuralCRF(nn.Module):
	def __init__(self, num_tag, hidden_size=100, num_layers=1, topk=30, num_units=200, use_cuda=False):
		super(NeuralCRF, self).__init__()
		self.num_layers = num_layers
		self.num_units = 2 * num_units
		self.num_tag = num_tag
		self.y0 = num_tag
		self.topk = topk
		self.hidden_size = hidden_size
		self.use_cuda = use_cuda
		self.device = th.device('cuda' if use_cuda else 'cpu')
		self.embedding = nn.Embedding.from_pretrained(th.eye(num_tag + 1, dtype=th.float32), freeze=True)
		self.nll_loss = nn.NLLLoss()
		self.gru = nn.GRU(num_tag + 1, hidden_size, batch_first=True)
		self.linear = nn.Linear(hidden_size, num_tag)
		
		self.jt_nn = nn.Linear(self.num_units + hidden_size, hidden_size)
		self.dense = nn.Linear(hidden_size, num_tag)
	
	def init_hidden(self, dtype, size):
		h = th.zeros(size, dtype=dtype, device=self.device)
		return h
	
	def init_label(self, dtype, size):
		x = th.zeros(size, dtype=dtype, device=self.device)
		x.fill_(self.y0)
		return x.long()
	
	def forward(self, emissions, tags, mask):
		"""
		crf 极大似然估计计算损失函数 w*f - logz
		:param emissions: 发射概率
		:param tags: 标签
		:param mask: pad
		:return: log(p)
		"""
		mask = mask.float()
		batch_size, seq_len, num_tag = emissions.size()
		path_score = self.path_score(emissions, tags, batch_size, seq_len, mask)
		with th.no_grad():
			search_results, search_bps, search_scores = self.beam_search_decode(emissions, batch_size, seq_len,
																				mask)
		search_tags, path = self.search_backward(search_results, search_bps, search_scores, mask)
		beam_scores = self.beam_search_scores(emissions, search_tags, batch_size, seq_len, mask)
		loss, _, _ = self.loss(path_score, tags, beam_scores, search_tags, batch_size, seq_len, mask)
		return loss, path
	
	def decode(self, emissions, mask):
		"""
		crf 极大似然估计计算损失函数 w*f - logz
		:param emissions: 发射概率
		:param mask: pad
		:return: log(p)
		"""
		mask = mask.float()
		batch_size, seq_len, num_tag = emissions.size()
		search_results, search_bps, search_scores = self.beam_search_decode(emissions, batch_size, seq_len, mask)
		search_tags, path = self.search_backward(search_results, search_bps, search_scores, mask)
		return path
	
	def loss(self, path_score, target, search_score, search_target, batch_size, seq_len, mask):
		"""
		损失函数
		:param path_score: [batch*seq_len, num_tag]
		:param target: [batch ,seq_len]
		:param search_score: [batch*topk*seq_len, num_tag]
		:param search_target: [batch,topk,seq_len]
		:param batch_size:
		:param seq_len:
		:param mask: batch seq_len
		:return:
		"""
		target = target.view(-1)
		search_target = search_target.view(-1)
		
		sample_mask = mask.repeat(1, self.topk).view(batch_size * self.topk, -1)
		
		# sum(score[i] * y[i]) ==> batch * seq_len
		data_loss = -self.nll_loss(path_score, target) * mask.view(-1)
		
		# batch topk seq_len
		sample_loss = -self.nll_loss(search_score, search_target) * sample_mask.contiguous().view(-1)
		
		data_loss = data_loss.contiguous().view(batch_size, 1, seq_len)
		sample_loss = sample_loss.contiguous().view(batch_size, self.topk, seq_len)
		sample_loss = th.cat((data_loss, sample_loss), dim=1)
		data_loss = data_loss.sum() / batch_size
		# 路径分数之和 logz =log( sum(exp(sum(path_score))) )
		sample_loss = th.logsumexp(th.sum(sample_loss, dim=2), dim=1)
		sample_loss = th.mean(sample_loss, dim=0)
		# 损失函数 负对数似然函数 -log(exp(path_score)/z) = logz - path_score
		loss = sample_loss - data_loss
		return loss, data_loss, sample_loss
	
	def beam_search_scores(self, emissions, search_tags, batch_size, seq_len, mask):
		
		emissions = emissions.contiguous().view(batch_size, -1).repeat(1, self.topk).view(-1, self.num_units)
		sos_mat = self.init_label(th.int64, (batch_size, self.topk, 1))
		
		x = th.cat((sos_mat, search_tags), dim=2)[:, :, :-1].view(-1, seq_len)
		input_embed = self.embedding(x)
		output, h = self.gru(input_embed)  # batch*topk seq_len hidden
		scores = self.joint_network(emissions,
									output.contiguous().view(-1, self.hidden_size))  # batch*topk*seq_len num_tag
		return scores
	
	def joint_network(self, trans_output, pred_output):
		projection = th.cat([trans_output, pred_output], dim=1)
		
		projection = th.tanh(self.jt_nn(projection))
		
		joint_output = self.dense(projection)
		
		return joint_output
	
	def path_score(self, emissions, tags, batch_size, seq_len, mask):
		"""
		路径分数
		:param emissions: batch seq_len hidden
		:param tags: batch seq_len
		:param batch_size:
		:param seq_len:
		:param mask:
		:return: w*f 真是路径特征分数
		"""
		sos_mat = self.init_label(th.int64, (batch_size, 1))
		
		x = th.cat((sos_mat, tags), dim=1)[:, :-1]
		input_embed = self.embedding(x)
		output, h = self.gru(input_embed)  # batch seq_len hidden
		# batch seq_len num_tag
		scores = self.joint_network(emissions.contiguous().view(-1, self.num_units),
									output.contiguous().view(-1, self.hidden_size))
		
		return scores
	
	def beam_search_decode(self, emissions, batch_size, seq_len, mask=None):
		x = self.init_label(th.int64, (batch_size * self.topk, 1))
		hx = self.init_hidden(th.float32, (self.num_layers, batch_size * self.topk, self.hidden_size))
		
		search_results = emissions.new_zeros((batch_size, self.topk, seq_len)).long()
		search_bps = emissions.new_zeros((batch_size, self.topk, seq_len)).long()
		# ignore the inflated copies to avoid duplicate entries in the top k
		search_scores = th.zeros((batch_size, self.topk, seq_len + 1), device=self.device)
		search_scores.fill_(-float('Inf'))
		search_scores.index_fill_(1, th.zeros((1), device=self.device).long(), 0.0)
		
		for step in range(seq_len):
			x = self.embedding(x)
			# batch*topk * 1 * hidden
			output, hx = self.gru(x, hx)
			
			emission = emissions[:, step, :].repeat(1, self.topk).contiguous().view(batch_size * self.topk, -1)
			
			score = self.joint_network(emission, output.squeeze(1))
			# 状态特征分数 + 转移特征分数
			score = score.contiguous().view(batch_size, self.topk, -1)  # batch k num_tag
			grow_score = search_scores[:, :, step].contiguous().view(batch_size, self.topk, 1) + score
			grow_score = grow_score.contiguous().view(batch_size, -1)  # batch * (topk*num_tag)
			# x,left_node 当前时刻输出  kp_beam 当前时刻输入 kp_score当前时刻最佳分数
			x, hx, kp_score, left_node, kp_beam = self.decision_step(batch_size, grow_score, hx)
			
			search_results[:, :, step] = left_node  # t时刻输出id
			search_bps[:, :, step] = kp_beam  # t时刻输入id
			search_scores[:, :, step + 1] = kp_score
		return search_results, search_bps, search_scores
	
	def search_backward(self, left_nodes, kp_beams, kp_scores, mask):
		
		"""
		args:
			left_nodes: int64 Tensor with shape [batch, num_samples, seq_len], where seq_len is the maximum of the true length in this minibatch.
			kp_beams: int64 Tensor with shape [batch, num_samples, seq_len]. 前一时刻输出的索引
			kp_scores: float32 Tensor with shape [batch, num_samples, seq_len+1].
			mask: int64 Tensor with shape [batch,seq_len].

		return :
			search_results: int64 Tensor with shape [batch, num_samples, seq_len].
		"""
		length = mask.long().sum(1)
		batch_size, num_samples, seq_lens = left_nodes.size()
		
		search_results = left_nodes.new_zeros((batch_size, num_samples, seq_lens)).long()
		path = []
		for n in range(batch_size):
			
			t = length[n] - 1  # 句子长度
			last_index = th.arange(num_samples).to(self.device).long()
			
			search_results[n, :, t] = left_nodes[n, :, t]
			
			ancestor_index = last_index
			# 反向追踪路径 kp_beams 前一时刻的输出索引
			for j in range(length[n] - 2, -1, -1):
				ancestor_index = th.index_select(kp_beams[n, :, j + 1], 0, ancestor_index)
				search_results[n, :, j] = th.index_select(left_nodes[n, :, j], 0, ancestor_index)
			path.append(search_results[n, 0, :length[n]].cpu().tolist())
		return search_results, path
	
	def decision_step(self, batch_size, score, hx):
		"""
		排序 从topk* num_tag中选取topk个最大值
		:param batch_size:
		:param score:
		:param hx:
		:return:
		"""
		topv, topi = th.topk(score, self.topk)  # topk*num_tag ->topk
		class_id = th.fmod(topi, self.num_tag).long()  # 求余数 当前时刻的输出状态
		beam_id = th.div(topi, self.num_tag).long()  # 求整数 前一时刻输出的索引
		
		x = class_id.contiguous().view(-1, 1)
		
		batch_offset = (th.arange(batch_size) * self.topk).view(batch_size, 1).to(self.device).long()
		
		# 前一时刻的隐状态
		state_id = batch_offset + beam_id
		state_id = state_id.view(-1)
		hx = th.index_select(hx, 1, state_id).view(-1, batch_size * self.topk, self.hidden_size)
		
		return x, hx, topv, class_id, beam_id


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
