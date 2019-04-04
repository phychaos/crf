#!/usr/bin python3
# -*- coding: utf-8 -*-
# @时间   : 19-4-4 下午3:04
# @作者   : Lin lifang
# @文件   : crf.py

import torch.nn as nn
import torch as th


class CRF(nn.Module):
	def __init__(self, num_tag):
		super(CRF, self).__init__()
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
		return likelihood,max_score, max_score_pre

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
