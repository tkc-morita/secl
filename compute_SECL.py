# coding: utf-8

import numpy as np
import pandas as pd
import argparse, os.path, glob



def get_loss_diff(df_full, df_local):
	df_full = df_full.sort_values('seq_and_time_ixs')
	for context_length, sub_df_local in df_local.groupby('context_length'):
		sub_df_local = sub_df_local.sort_values('seq_and_time_ixs')
		df_local.loc[sub_df_local.index, 'loss_diff'] = sub_df_local.loss - df_full.loss.values
	df_local.loc[:,'context_length'] = df_local.context_length.astype(int)
	return df_local


def get_statistically_effective_context_length(
		df,
		group_name,
		threshold=0.01,
		seed=111,
		num_bs_samples=10000
	):
	update_message = """
	UPDATE on Jun 25 2021:
	TO make it parallel to the Markovian order, SECL is now defined as
	"the MINIMUM length of truncated context where its difference from the full context is BELOW the threshold."
	Previously, the definition was:
	"the MAXIMUM length of truncated context where its difference from the full context is ABOVE the threshold."
	In short, this revision makes the new SECL = old SECL + 1.
	"""
	print(update_message)
	ecls = {}
	for gp, sub_df in df.groupby(group_name):
		print('{}:{}'.format(group_name, gp))
		# UPDATE on Jun 25 2021:
		# TO make it parallel to the Markovian order, SECL is now defined as "the minimum "
		# previous_context_length = 0
		# previous_perplex_gain = None
		ecl_detected = False
		for context_length, subsub_df in sub_df.groupby('context_length'):
			random_state = np.random.RandomState(seed)
			samples = [subsub_df.sample(frac=1.0, replace=True, random_state=random_state).loss_diff.mean()
				for iter_ix in range(num_bs_samples)
			]
			perplex_gain = np.exp(np.percentile(samples,5.0))
			if perplex_gain < 1.0+threshold:
				print('N={}'.format(subsub_df.shape[0]))
				print('Statistically Effective Context Length (SECL) is {}'.format(context_length))
				print('Perplexity improvement at SECL: {}'.format(perplex_gain))
				# print('Perplexity improvement at {}: {}'.format(context_length, perplex_gain))
				ecls[gp] = context_length
				ecl_detected = True
				break
			# else:
				# previous_context_length = context_length
				# previous_perplex_gain = perplex_gain
		if not ecl_detected:
			print('N={}'.format(subsub_df.shape[0]))
			print('Statistically Effective Context Length (SECL) is >{}'.format(context_length))
			print('Perplexity improvement at SECL: {}'.format(perplex_gain))
			print('Achieved the maximum tested.')
			ecls[gp] = context_length
	ecls = pd.Series(ecls, name='SECL')
	print(ecls.describe())
	return ecls

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', '--full_context', type=str, help='Path to csv containing full context results.')
	parser.add_argument('-l', '--local_context', nargs='+', type=str, default=[], help='Space-separated list of paths to local context results, headed by the context length.')
	parser.add_argument('-S', '--save_path', type=str, default='SECL.csv', help='Path to the csv where results are saved.')
	parser.add_argument('--sep', default=',', type=str, help='Separator symbol of the data files.')
	parser.add_argument('--annotation_col', type=str, default=None, help='Name of the column for the annotation info.')
	parser.add_argument('--new_annotation_name', type=str, default=None, help='Rename the column for the annotation info.')
	parser.add_argument('--rename_annotations', type=str, default=None, help='Path to the json file containing the renaming rule.')
	args = parser.parse_args()

	dfs = []
	rename_columns = {
		# "bout_ix":"seq_ix",
		# "time_ix_in_bout":"time_ix",
		"sentence_id":"seq_ix",
		"ID_contiguous":"time_ix",
		"subfile":"seq_ix",
		"time_ix_in_subfile":"time_ix"
	}
	if args.sep in ['t','\\t']:
		args.sep = '\t'

	cols = ['seq_ix','time_ix','speaker','loss']
	if not args.annotation_col is None:
		cols.append(args.annotation_col)

	df_full = pd.read_csv(args.full_context, sep=args.sep, encoding='utf-8').rename(columns=rename_columns)
	if not 'speaker' in df_full.columns:
		df_full['speaker'] = 'NA'
	df_full = df_full.loc[:,cols]
	df_full['context_length'] = 'full'
	df_full['seq_and_time_ixs'] = df_full.seq_ix.astype(int).astype(str) + '_' + df_full.time_ix.astype(int).astype(str)

	dfs = []
	for locality, path in zip(args.local_context[0::2], args.local_context[1::2]):
		sub_df = pd.read_csv(path, sep=args.sep, encoding='utf-8').rename(columns=rename_columns)
		if not 'speaker' in sub_df.columns:
			sub_df['speaker'] = 'NA'
		sub_df = sub_df.loc[:,cols]
		sub_df['context_length'] = locality
		dfs.append(sub_df)
	df_local = pd.concat(dfs, axis=0, ignore_index=True)
	df_local['seq_and_time_ixs'] = df_local.seq_ix.astype(int).astype(str) + '_' + df_local.time_ix.astype(int).astype(str)


	# For vs. human comparison
	new_dfs_full = []
	new_dfs_local = []
	data2counts = df_local.seq_and_time_ixs.value_counts()
	target_data_ixs = data2counts[data2counts==data2counts.max()].index # Only use data included in all the files.
	df_full = df_full[df_full.seq_and_time_ixs.isin(target_data_ixs)]
	df_local = df_local[df_local.seq_and_time_ixs.isin(target_data_ixs)]

	df = get_loss_diff(df_full, df_local)

	if args.annotation_col=='FEATS':
		target_feature = 'PronType=Int'
		df[target_feature] = df['FEATS'].str.contains(target_feature)
		args.annotation_col = target_feature
	elif args.annotation_col is None:
		args.annotation_col = 'dummy_col'
		df[args.annotation_col] = 'all'
	elif not args.rename_annotations is None:
		import json
		with open(args.rename_annotations, 'r') as f:
			rename_annotations = json.load(f)
		df.loc[:,args.annotation_col] = df.loc[:,args.annotation_col].astype(str).map(rename_annotations)

	if not args.new_annotation_name is None:
		df.loc[:,args.new_annotation_name] = df[args.annotation_col]
		args.annotation_col = args.new_annotation_name


	secls = get_statistically_effective_context_length(df, args.annotation_col)
	secls.to_frame().reset_index().rename(columns={'index':args.annotation_col}).to_csv(args.save_path, index=False)