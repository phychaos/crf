#!/usr/bin python3
# -*- coding: utf-8 -*-
# @时间   : 19-4-4 下午3:04
# @作者   : Lin lifang
# @文件   : crf.py

import torch as th
from typing import List, Optional, Union
from torch.autograd import Variable
import torch.nn as nn

class CRF1(nn.Module):
	def __init__(self, num_tag):
		super(CRF1, self).__init__()
		self.num_tag = num_tag
		self.transitions = nn.Parameter(th.ones(num_tag, num_tag))

	def forward(self, x, mask=None):
		"""
		:param x:
		:param mask:
		:return:
		"""
		batch_size, seq_len, num_tag = x.size()
		transitions = self.transitions.unsqueeze(dim=0).repeat((batch_size, 1, 1))
		alphas, log_z, max_score, max_score_pre = self.forward_alpha(x, transitions, batch_size, seq_len, mask)
		target_score = x.sum() + self.transitions.sum()
		likelihood = target_score - log_z
		return likelihood, max_score, max_score_pre

	def forward_alpha(self, observations, transitions, batch_size, seq_len, mask=None):
		"""
		前向算法 alpha  = dot(alpha, M) = exp(log alpha + log M)
		:param observations:
		:param transitions:
		:param batch_size:
		:param seq_len:
		:param mask:
		:return:
		"""
		previous = observations[:, 0, :] + transitions[:, 0, :]
		previous = previous.view((batch_size, self.num_tag))
		alphas = [previous]
		max_score = [previous]
		max_score_pre = [th.range(0, self.num_tag - 1, 1)]
		for t in range(1, seq_len):
			previous = previous.view((batch_size, self.num_tag, 1))
			log_m = observations[:, t, :].view((batch_size, self.num_tag, 1)) + transitions
			alpha_t = previous + log_m
			max_score.append(alpha_t.max(dim=1))
			max_score_pre.append(th.argmax(alpha_t, dim=1))
			alpha_t = alpha_t.logsumexp(dim=1, keepdim=False)

			alphas.append(alpha_t)
			previous = alpha_t
			alpha_t.logsumexp()
		alphas = th.cat(tuple(alphas), dim=0).transpose(0, 1)  # batch * seq_len * num_tag
		if mask:
			_index = th.Tensor([[kk, ii, range(self.num_tag)] for kk, ii in enumerate(range(mask))])
			log_z = alphas.gather(dim=0, index=_index)
		else:
			log_z = previous
		log_z = log_z.logsumexp(dim=1).sum()
		max_score = th.cat(tuple(max_score), dim=0).transpose(0, 1)
		max_score_pre = th.cat(tuple(max_score_pre), dim=0).transpose(0, 1)
		return alphas, log_z, max_score, max_score_pre

	def viterbi(self, max_scores, max_scores_pre, mask):
		"""
		维特比反向算法
		:param max_scores:
		:param max_scores_pre:
		:param mask:
		:return:
		"""
		batch_size, _, num_tag = max_scores.size()
		best_paths = []
		for ii in range(batch_size):
			path = []
			last_max_node = th.argmax(max_scores[ii, mask[ii] - 1])
			path.append(last_max_node)
			for t in range(mask[ii], 0, -1):
				last_max_node = max_scores_pre[ii, t, last_max_node]
				path.append(last_max_node)
			path = path[::-1]
			best_paths.append(path)
		return best_paths


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
