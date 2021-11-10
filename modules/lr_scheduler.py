# coding: utf-8

import torch

class LinearWarmUp(torch.optim.lr_scheduler._LRScheduler):
	''''''
	def __init__(self, optimizer, warmup_iters, last_epoch=-1):
		self.warmup_iters = warmup_iters
		super(LinearWarmUp, self).__init__(optimizer, last_epoch)

	def get_lr(self):
		lr_scale = self._get_lr_scale()
		return [
			base_lr * lr_scale
			for base_lr in self.base_lrs
		]

	def _get_lr_scale(self):
		"""
		Based on BERT-pytorch
		https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/trainer/optim_schedule.py
		"""
		if self.last_epoch < self.warmup_iters:
			return (self.warmup_iters**-1.5) * (self.last_epoch+1)
		else:
			return (self.last_epoch+1)**-0.5
