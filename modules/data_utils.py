# coding: utf-8

import torch
import torch.utils.data
import pandas as pd
import numpy as np
import os.path
import warnings
warnings.simplefilter("error")


class Dataset(torch.utils.data.Dataset):
	def __init__(
			self,
			data_path,
			transform = None,
			normalize_by_max_L2 = False,
			seq_col='seq_ix',
			time_col='time_ix',
			dim_col='dim',
			val_col='val',
			prob_col='prob',
			individual_col=None,
			discrete=False,
			noisy_data=False,
			backward_modeling=False,
			label2ix=None,
			individual2ix=None,
			sep=','
			):
		if sep in ['t','\\t']:
			sep = '\t'
		self.df_data = pd.read_csv(data_path, sep=sep)
		self.seq_col = seq_col
		self.time_col = time_col
		self.dim_col = dim_col
		self.val_col = val_col
		self.prob_col = prob_col
		self.discrete = discrete
		self.noisy_data = noisy_data
		self.backward_modeling = backward_modeling
		self.df_data.loc[:,self.val_col] = self.df_data[self.val_col].astype(str)
		self.df_seq = self.df_data.groupby(seq_col)[time_col].max().to_frame().rename(columns={time_col:'length'}).reset_index()
		self.df_seq.loc[:,'length'] += 1
		self.transform = transform
		self.normalize_by_max_L2 = normalize_by_max_L2
		if normalize_by_max_L2:
			self.normalizer = self.get_max_L2_normalizer()
		self.individual_col = individual_col
		if not individual_col is None:
			if individual2ix is None:
				self.individual2ix = {indiv:ix for ix,indiv in enumerate(self.df_data[individual_col].unique())}
			else:
				self.individual2ix = individual2ix
			self.ix2individual = {ix:indiv for indiv,ix in self.individual2ix.items()}
		if self.discrete:
			if label2ix is None:
				label2ix = {label:ix for ix,label in enumerate(self.df_data[self.val_col].unique(), 1)}
				label2ix['<sos>'] = 0
			self.set_label2ix(label2ix)
		if self.noisy_data:
			self.random_state = np.random.RandomState()

	def get_max_L2_normalizer(self):
		data = self.df_data.sort_values(
					[self.seq_col,self.time_col,self.dim_col]
				)[self.val_col].values.reshape(
					-1, self.get_input_size()
				)
		return np.power(np.power(data, 2.0).sum(-1), 0.5).max()

	def get_num_individuals(self):
		if self.individual_col is None:
			return None
		return len(self.individual2ix)

	def set_label2ix(self, label2ix):
		self.label2ix = label2ix
		self.ix2label = {ix:label for label,ix in self.label2ix.items()}

	def activate_L2_normalizer(self):
		self.normalize_by_max_L2 = True
		self.normalizer = self.get_max_L2_normalizer()

	def __len__(self):
		"""Return # of data strings."""
		return self.df_seq.shape[0]

	def get_input_size(self):
		if self.discrete:
			return len(self.label2ix)
		else:
			return len(self.df_data[self.dim_col].unique())

	def get_quantiles(self, qs):
		return self.df_data.groupby(self.dim_col, sort=True).quantile(qs).unstack(level=1).values


	def seed_random_state(self, seed=111):
		if self.noisy_data:
			seed = seed % 2**32
			self.random_state.seed(seed=seed)

	def __getitem__(self, ix):
		"""Return """
		seq_ix = self.df_seq.loc[ix, self.seq_col]

		df_seq = self.df_data.loc[self.df_data[self.seq_col]==seq_ix,:]
		if self.discrete:
			if self.noisy_data:
				df_seq = df_seq.groupby(self.time_col, sort=True
							).apply(lambda df_time: df_time.sample(
									n=1, weights=self.prob_col,
									random_state=self.random_state
							)).reset_index(level=self.time_col, drop=True)
			else:
				df_seq = df_seq.sort_values([self.time_col])
			target_data = df_seq[self.val_col].map(self.label2ix).values.astype(int)
			init_filler = self.label2ix['<sos>']
			time_axis = None
		else:
			df_seq = df_seq.sort_values([self.time_col, self.dim_col])
			target_data = df_seq.pivot(index=self.time_col, columns=self.dim_col, values=self.val_col).values.astype(np.float32)
			init_filler = np.zeros((1,target_data.shape[1]), dtype=np.float32)
			time_axis = 0
		if self.backward_modeling:
			target_data = target_data[::-1].copy() # Negative stride not supported by PyTorch, so need to create a new array.
		input_data = np.append(init_filler, target_data[:-1], axis=time_axis)
		input_data = torch.from_numpy(input_data)
		target_data = torch.from_numpy(target_data)

		if self.individual_col is None:
			individual = 0.0 / torch.zeros(target_data.size(0))
		else:
			individual = df_seq.drop_duplicates(subset=[self.time_col])[self.individual_col].map(self.individual2ix).values
			individual = torch.from_numpy(individual)
		if self.normalize_by_max_L2:
			input_data /= self.normalizer
			target_data /= self.normalizer
		if self.transform:
			input_data, target_data = self.transform(input_data, target_data)
		return input_data, target_data, individual, ix


class Transform(object):
	def __init__(self, in_trans, out_trans):
		self.in_trans = in_trans
		self.out_trans = out_trans
		
	def __call__(self, input_data, target_data):
		in_transformed = self.in_trans(input_data)
		target_transformed = self.out_trans(target_data)
		return in_transformed, target_transformed


class Compose(object):
	def __init__(self, transforms):
		self.transforms = transforms

	def __call__(self, *data):
		for trans in self.transforms:
			data = trans(*data)
		return data

class IterationBasedBatchSampler(torch.utils.data.BatchSampler):
	"""
	Wraps a BatchSampler, resampling from it until
	a specified number of iterations have been sampled.
	Partially Copied from maskedrcnn-benchmark.
	https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/data/samplers/iteration_based_batch_sampler.py
	"""

	def __init__(self, batch_sampler, num_iterations, start_iter=0):
		self.batch_sampler = batch_sampler
		self.num_iterations = num_iterations
		self.start_iter = start_iter
		if hasattr(self.batch_sampler.sampler, 'set_start_ix'):
			start_ix = (self.start_iter % len(self.batch_sampler)) * self.batch_sampler.batch_size
			self.batch_sampler.sampler.set_start_ix(start_ix)

	def __iter__(self):
		iteration = self.start_iter
		epoch = iteration // len(self.batch_sampler)
		while iteration <= self.num_iterations:
			if hasattr(self.batch_sampler.sampler, 'set_epoch'):
				self.batch_sampler.sampler.set_epoch(epoch)
			for batch in self.batch_sampler:
				iteration += 1
				if iteration > self.num_iterations:
					break
				yield batch
			epoch += 1

	def __len__(self):
		return self.num_iterations

class RandomSampler(torch.utils.data.RandomSampler):
	"""
	Custom random sampler for iteration-based learning.
	"""
	def __init__(self, *args, seed=111, **kwargs):
		super(RandomSampler, self).__init__(*args, **kwargs)
		self.epoch = 0
		self.start_ix = 0
		self.seed = seed

	def set_epoch(self, epoch):
		self.epoch = epoch

	def set_start_ix(self, start_ix):
		self.start_ix = start_ix

	def __iter__(self):
		g = torch.Generator()
		g.manual_seed(self.epoch+self.seed)
		start_ix = self.start_ix
		self.start_ix = 0
		n = len(self.data_source)
		if self.replacement:
			return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64, generator=g).tolist()[start_ix:])
		return iter(torch.randperm(n, generator=g).tolist()[start_ix:])


def get_data_loader(dataset, batch_size=1, shuffle=False, num_iterations=None, start_iter=0, num_workers=1, random_seed=111):
	if shuffle:
		sampler = RandomSampler(dataset, replacement=False, seed=random_seed)
	else:
		sampler = torch.utils.data.SequentialSampler(dataset)
	batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=False)
	if not num_iterations is None:
		batch_sampler = IterationBasedBatchSampler(batch_sampler, num_iterations, start_iter=start_iter)

	data_loader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_sampler=batch_sampler, collate_fn=to_packed_sequence, worker_init_fn=worker_init_fn)
	return data_loader

def to_packed_sequence(batch):
	batch = sorted(batch, key=lambda seq_and_ix: seq_and_ix[0].size(0), reverse=True) # Reordering
	input_seqs, target_seqs, indivs, ixs = zip(*batch)
	input_seqs = torch.nn.utils.rnn.pack_sequence(input_seqs)
	target_seqs = torch.nn.utils.rnn.pack_sequence(target_seqs)
	indivs = torch.nn.utils.rnn.pack_sequence(indivs)
	return input_seqs, target_seqs, indivs, ixs


def worker_init_fn(worker_ix):
	worker_info = torch.utils.data.get_worker_info()
	dataset = worker_info.dataset  # the dataset copy in this worker process
	dataset.seed_random_state(worker_info.seed)