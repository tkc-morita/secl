# encoding: utf-8

import torch
from modules.data_utils import Compose
import numpy as np
import pandas as pd
from modules import data_utils, model
import learning
import os, argparse, itertools


class Predictor(learning.Learner):
	def __init__(self, model_config_path, device = 'cpu'):
		self.device = torch.device(device)
		self.retrieve_model(checkpoint_path = model_config_path, device=device)
		for param in self.parameters():
			param.requires_grad = False
		[m.eval() for m in self.modules]


	def predict(self, input_data, target_data, individuals, is_packed = False, is_flatten=False, batch_sizes=None, to_numpy = True):
		if is_packed:
			batch_sizes = input_data.batch_sizes
			input_data = input_data.data
			target_data = target_data.data
			individuals = individuals.data
		elif not is_flatten:
			if not isinstance(input_data, list):
				input_data = [input_data]
			if not isinstance(target_data, list):
				target_data = [target_data]
			if not isinstance(individuals, list):
				individuals = [individuals]
			input_data = torch.nn.utils.rnn.pack_sequence(input_data)
			batch_sizes = input_data.batch_sizes
			input_data = input_data.data
			target_data = torch.nn.utils.rnn.pack_sequence(target_data).data
			individuals = torch.nn.utils.rnn.pack_sequence(individuals).data
		with torch.no_grad():
			input_data = input_data.to(self.device)
			target_data = target_data.to(self.device)
			individuals = individuals.to(self.device)
			att,_ = self.attention(input_data, individuals=individuals, batch_sizes=batch_sizes) # For some reason, passing data directly to the network automatically converts it into a tuple of data's attributes.
			loss = self.to_predictions_and_loss(att, target_data)
		if to_numpy:
			loss = loss.data.cpu().numpy()
		return loss


	def predict_dataset(
			self,
			dataset,
			save_path,
			to_numpy = True,
			batch_size=1,
			num_workers=1,
			seq_col='seq_ix',
			time_col='time_ix',
			dim_col='dim',
			val_col='val',
			locality=None,
			sep=','
			):
		if sep in ['t','\\t']:
			sep = '\t'
		dataloader = data_utils.get_data_loader(
			dataset,
			batch_size=batch_size,
			shuffle=False,
			num_workers=num_workers
			)
		rename_existing_file(save_path)
		sub_df_data = dataset.df_data.drop_duplicates(subset=[seq_col,time_col])
		if not self.discrete:
			sub_df_data = sub_df_data.drop(columns=[dim_col,val_col])
		# results = []
		for input_data, target_data, individuals, ix_in_list in dataloader:
			batch_boundaries = [0]+target_data.batch_sizes.cumsum(0).tolist()
			batch_start = batch_boundaries[:-1]
			batch_end = batch_boundaries[1:]
			if locality is None:
				loss = self.predict(input_data, target_data, individuals, is_packed=True, to_numpy=to_numpy)
				results = [(data_ix_,time_ix_,l)
							for time_ix_, (bs,be) in enumerate(zip(batch_start,batch_end))
							for data_ix_,l in zip(ix_in_list, loss[bs:be])
							]
			else:
				assert locality>0, 'locality must be at least 1.'
				results = []
				for target_time_ix_plus_one in range(locality,len(batch_boundaries)):
					context_onset = target_time_ix_plus_one - locality
					onset_in_flatten = batch_boundaries[context_onset]
					offset_in_flatten = batch_boundaries[target_time_ix_plus_one]
					sub_input = input_data.data[onset_in_flatten:offset_in_flatten]
					sub_target = target_data.data[onset_in_flatten:offset_in_flatten]
					sub_batch_sizes = input_data.batch_sizes[context_onset:target_time_ix_plus_one]
					sub_individuals = individuals.data[onset_in_flatten:offset_in_flatten]
					loss = self.predict(sub_input, sub_target, sub_individuals, is_flatten=True, batch_sizes=sub_batch_sizes, to_numpy=to_numpy)
					loss = loss[-sub_batch_sizes[-1]:]
					results += [(data_ix_,target_time_ix_plus_one-1,l) for data_ix_,l in zip(ix_in_list, loss)]
			if results:
				df_pred = pd.DataFrame(
								results,
								columns=['data_ix_att',time_col,'loss']
							)
				df_pred = df_pred.merge(
					dataset.df_seq,
					how='left',
					left_on='data_ix_att',
					right_index=True
					).drop(columns=['data_ix_att'])
				df_pred = df_pred.merge(
							sub_df_data,
							how='left',
							on=[seq_col,time_col]
							)
				if os.path.isfile(save_path):
					df_pred.to_csv(save_path, index=False, mode='a', header=False, sep=sep)
				else:
					df_pred.to_csv(save_path, index=False, sep=sep)

	@staticmethod
	def get_expected_dependency_lengths(weight_mat):
		pos = np.arange(weight_mat.shape[0])
		lengths = np.absolute(pos.reshape(-1,1) - pos.reshape(1,-1))
		expected_lengths = (lengths * weight_mat).sum(-1)
		return expected_lengths

	@staticmethod
	def get_entropy(weight_mat):
		base = weight_mat.shape[0]
		entropy = -(weight_mat * np.ma.log(weight_mat)).sum(-1)
		if base>1:
			entropy /= np.log(base)
		return entropy
	
def rename_existing_file(filepath):
	if os.path.isfile(filepath):
		new_path = filepath+'.orig'
		rename_existing_file(new_path)
		os.rename(filepath, new_path)

def get_parameters():
	parser = argparse.ArgumentParser()
	
	parser.add_argument('model_path', type=str, help='Path to the configuration file of a trained model.')
	parser.add_argument('data_path', type=str, help='Path to the csv file containing the test data.')
	parser.add_argument("--seq_col", type=str, default='sequence_ix', help="Name of the csv column containing the index of the sequences.")
	parser.add_argument("--time_col", type=str, default='time_ix', help="Name of the csv column containing the discrete time indexes of the sequences.")
	parser.add_argument("--dim_col", type=str, default='dim', help="Name of the csv column containing the index of the dimensions of the data points.")
	parser.add_argument("--val_col", type=str, default='value', help="Name of the csv column containing the data values.")
	parser.add_argument('--individual_col', type=str, default=None, help='Name of the csv column containing the individual info.')
	parser.add_argument('--sep', type=str, default=',', help='The separator of the data')
	parser.add_argument('-d', '--device', type=str, default='cpu', help='Computing device.')
	parser.add_argument('-S', '--save_path', type=str, default=None, help='Path to the csv where results are saved.')
	parser.add_argument('-b', '--batch_size', type=int, default=1, help='Batch size.')
	parser.add_argument('--num_workers', type=int, default=0, help='# of workers for dataloading.')
	parser.add_argument('--locality', type=int, default=None, help='If specified, attention is limited to neighbors whose distance threshold is determined by this value.')
	parser.add_argument('--backward_modeling', action='store_true', help='If selected, language modeling runs backward, from future to past.')

	return parser.parse_args()

if __name__ == '__main__':
	args = get_parameters()

	save_path = args.save_path
	if save_path is None:
		save_path = os.path.join(os.path.dirname(args.input_root), 'loss.csv')
	if not os.path.isdir(os.path.dirname(save_path)):
		os.makedirs(os.path.dirname(save_path))

	# Get a model.
	predictor = Predictor(args.model_path, device=args.device)
	if predictor.discrete:
		import json
		with open(os.path.join(os.path.dirname(args.model_path), 'label_coding.json'), 'r') as f:
			label2ix = json.load(f)
	else:
		label2ix = None
	individual2ix_path = os.path.join(os.path.dirname(args.model_path), 'individual_coding.json')
	if os.path.isfile(individual2ix_path):
		import json
		with open(individual2ix_path, 'r') as f:
			individual2ix = json.load(f)
	else:
		individual2ix = None

	dataset = data_utils.Dataset(
		args.data_path,
		seq_col=args.seq_col,
		time_col=args.time_col,
		dim_col=args.dim_col,
		val_col=args.val_col,
		normalize_by_max_L2=predictor._normalize_input_max_L2,
		individual_col=args.individual_col,
		discrete=predictor.discrete if hasattr(predictor, 'discrete') else False,
		backward_modeling=args.backward_modeling,
		label2ix=label2ix,
		individual2ix=individual2ix,
		sep=args.sep,
	)


	predictor.predict_dataset(
		dataset,
		save_path,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		seq_col=args.seq_col,
		time_col=args.time_col,
		dim_col=args.dim_col,
		val_col=args.val_col,
		locality=args.locality,
		sep=args.sep
	)
