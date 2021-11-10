# coding: utf-8

import torch
import numpy as np
from modules import model, data_utils, lr_scheduler
from logging import getLogger,FileHandler,DEBUG,Formatter
import os, argparse, itertools, json

logger = getLogger(__name__)

def update_log_handler(file_dir):
	current_handlers=logger.handlers[:]
	for h in current_handlers:
		logger.removeHandler(h)
	log_file_path = os.path.join(file_dir,'history.log')
	if os.path.isfile(log_file_path):
		retrieval = True
	else:
		retrieval = False
	handler = FileHandler(filename=log_file_path)	#Define the handler.
	handler.setLevel(DEBUG)
	formatter = Formatter('{asctime} - {levelname} - {message}', style='{')	#Define the log format.
	handler.setFormatter(formatter)
	logger.setLevel(DEBUG)
	logger.addHandler(handler)	#Register the handler for the logger.
	if retrieval:
		logger.info("LEARNING RETRIEVED.")
	else:
		logger.info("Logger set up.")
		logger.info("PyTorch ver.: {ver}".format(ver=torch.__version__))
	return retrieval,log_file_path



class Learner(object):
	def __init__(self,
			input_size,
			save_dir,
			attention_hidden_size = 512,
			num_attention_heads = 8,
			num_attention_layers = 1,
			bottleneck_layers=list(),
			dropout= 0.0,
			gauss_mix_loss=False,
			dirichlet_loss=False,
			cross_entropy_loss=False,
			num_predictions=1,
			locality=None,
			sparsemax=False,
			num_individuals=None,
			discrete=False,
			quantized_output=False,
			quantization_boundaries=None,
			device='cpu',
			seed=1111
			):
		self.retrieval,self.log_file_path = update_log_handler(save_dir)
		self.device = torch.device(device)
		logger.info('Device: {device}'.format(device=device))
		self.distributed = False
		if torch.cuda.is_available():
			if device.startswith('cuda'):
				logger.info('CUDA Version: {version}'.format(version=torch.version.cuda))
				if torch.backends.cudnn.enabled:
					logger.info('cuDNN Version: {version}'.format(version=torch.backends.cudnn.version()))
				# torch.distributed.init_process_group('nccl', rank=0, world_size=1) # Currently only support single-process with multi-gpu situation.
				self.distributed = True
			else:
				print('CUDA is available. Restart with option -C or --cuda to activate it.')

		self.save_dir = save_dir

		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

		if self.retrieval:
			self.last_iteration = self.retrieve_model(device=device)
			logger.info('Model retrieved.')
		else:
			self.seed = seed
			torch.manual_seed(seed)
			torch.cuda.manual_seed_all(seed) # According to the docs, "It’s safe to call this function if CUDA is not available; in that case, it is silently ignored."
			logger.info('Random seed: {seed}'.format(seed = seed))
			logger.info('# of attention layers: {}'.format(num_attention_layers))
			logger.info('# of attention hidden units per layer: {}'.format(attention_hidden_size))
			logger.info('# of attention heads: {}'.format(num_attention_heads))
			logger.info('Positions are relatively encoded by distance, with combination of sinusoid and learnable biases.')
			logger.info('Bottleneck layers where # of heads is 1 (ID# starts with 0): {}'.format(bottleneck_layers))
			logger.info('Dropout rate at the top of the sublayers and attention weights: {}'.format(dropout))
			if not locality is None:
				logger.info('Attention is limited to neighbors maximally {} time step(s) away.'.format(locality))
			logger.info('This is language modeling. Attention to future is prohibited.')
			if sparsemax:
				logger.info('Attention weights in the bottleneck layers are computed with sparsemax instead of softmax.')
				logger.info('Softmax is used for the non-bottleneck layers.')
			if not num_individuals is None:
				logger.info("Network receives embeddings of {} individuals' ID".format(num_individuals))
			self.bottleneck_layers = bottleneck_layers
			self.attention = model.SelfAttentionEncoder(
								input_size,
								attention_hidden_size,
								num_heads=num_attention_heads,
								num_layers=num_attention_layers,
								dropout=dropout,
								bottleneck_layers=bottleneck_layers,
								locality=locality,
								sparsemax=sparsemax,
								num_individuals=num_individuals,
								discrete=discrete
								)
			self.attention_init_args = self.attention.pack_init_args()
			if self.distributed:
				self.attention = torch.nn.DataParallel(self.attention)

			self.gauss_mix_loss = gauss_mix_loss
			self.dirichlet_loss = dirichlet_loss
			self.cross_entropy_loss = cross_entropy_loss
			self.discrete = discrete
			self.quantized_output = quantized_output
			if discrete:
				self.to_predictions_and_loss = model.DiscreteCrossEntropyLoss(attention_hidden_size, input_size)
				logger.info('Inputs are discrete labels ({} types).'.format(input_size))
				logger.info('loss: cross entropy')
			elif quantized_output:
				self.to_predictions_and_loss = model.CrossEntropyLossWithQuantization(attention_hidden_size, quantization_boundaries)
				logger.info('Outputs are quantized into {} levels per dim.'.format(quantization_boundaries.shape[1]+1))
				logger.info('loss: cross entropy')
			elif dirichlet_loss:
				self.to_predictions_and_loss = model.DirichletLoss(attention_hidden_size, input_size)
				logger.info('loss: negative log pdf of dirichlet')
			elif cross_entropy_loss:
				self.to_predictions_and_loss = model.InterProbCrossEntropyLoss(attention_hidden_size, input_size)
				logger.info('loss: cross entropy between the target and predicted categorical distributions')
			else:
				logger.info('# of predictions per data point: {}'.format(num_predictions))
				if gauss_mix_loss:
					self.to_predictions_and_loss = model.IsotropicGaussMixLoss(attention_hidden_size, input_size, num_predictions)
					logger.info('loss: negative log pdf of isotropic gaussian mixture')
				else:
					self.to_predictions_and_loss = model.MultiL2Loss(attention_hidden_size, input_size, num_predictions)
					logger.info('loss: MSE')
			
			self.modules = [self.attention, self.to_predictions_and_loss]
			[m.to(self.device) for m in self.modules]
			self.parameters = lambda:itertools.chain(*[m.parameters() for m in self.modules])



	def train(self, dataloader, saving_interval, start_iter=0):
		"""
		Training phase. Updates weights.
		"""
		[m.train() for m in self.modules]

		num_iterations = len(dataloader)
		total_loss = 0.0
		num_data_points_total = 0.0

		for iteration,(packed_input, packed_target, packed_individual, _) in enumerate(dataloader, start_iter):
			iteration += 1 # Original starts with 0.
			

			packed_input = packed_input.to(self.device)
			packed_target = packed_target.to(self.device)
			packed_individual = packed_individual.to(self.device)

			self.optimizer.zero_grad()
			torch.manual_seed(iteration+self.seed)
			torch.cuda.manual_seed_all(iteration+self.seed)

			out,weights = self.attention(packed_input.data, batch_sizes=packed_input.batch_sizes, individuals=packed_individual.data)
			loss = self.to_predictions_and_loss(out, packed_target.data)
			loss = loss.sum()

			total_loss += loss.item()
			num_data_points = packed_input.data.size(0)
			num_data_points_total += num_data_points
			loss /= num_data_points
			loss.backward()

			self.optimizer.step()
			self.lr_scheduler.step()


			if iteration % saving_interval == 0:
				logger.info('{iteration}/{num_iterations} iterations complete.'.format(iteration=iteration, num_iterations=num_iterations))
				if self.gauss_mix_loss or self.dirichlet_loss:
					logger.info('mean log prob. density (per data point): {loss}'.format(loss=-total_loss / num_data_points_total))
				elif self.cross_entropy_loss or self.discrete or self.quantized_output:
					logger.info('mean cross entropy loss (per data point): {loss}'.format(loss=total_loss / num_data_points_total))
				else:
					logger.info('mean L2 loss (per data point): {loss}'.format(loss=total_loss / num_data_points_total))
				total_loss = 0.0
				num_data_points_total = 0.0
				self.save_model(iteration-1, dataloader.dataset)
		self.save_model(iteration-1, dataloader.dataset)



	def learn(self, train_dataset, num_iterations, batch_size_train, learning_rate=1e-4, betas=(0.9, 0.999), decay=0.01, warmup_iters=0, saving_interval=200, num_workers=1):
		if self.retrieval:
			start_iter = self.last_iteration + 1
			logger.info('To be restarted from the beginning of iteration #: {iteration}'.format(iteration=start_iter+1))
			if self.discrete:
				train_dataset.set_label2ix(self._label2ix)
			if self._normalize_input_max_L2:
				train_dataset.activate_L2_normalizer()
			self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, betas=betas, weight_decay=decay)
			self.optimizer.load_state_dict(self.checkpoint['optimizer'])
			self.lr_scheduler = lr_scheduler.LinearWarmUp(self.optimizer, warmup_iters)
			self.lr_scheduler.load_state_dict(self.checkpoint['lr_scheduler'])
		else:
			self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, betas=betas, weight_decay=decay)
			self.lr_scheduler = lr_scheduler.LinearWarmUp(self.optimizer, warmup_iters)
			logger.info("START LEARNING.")
			logger.info("max # of iterations: {ep}".format(ep=num_iterations))
			logger.info("batch size for training data: {size}".format(size=batch_size_train))
			logger.info("initial learning rate: {lr}".format(lr=learning_rate))
			logger.info("weight decay: {decay}".format(decay=decay))
			logger.info("Betas: {betas}".format(betas=betas))
			logger.info("First {warmup_iters} iterations for warm-up.".format(warmup_iters=warmup_iters))
			start_iter = 0
		train_dataloader = data_utils.get_data_loader(
			train_dataset,
			batch_size=batch_size_train,
			start_iter=start_iter, 
			num_iterations=num_iterations,
			shuffle=True,
			num_workers=num_workers,
			random_seed=self.seed)
		self.train(train_dataloader, saving_interval, start_iter=start_iter)
		logger.info('END OF TRAINING')


	def save_model(self, iteration, dataset):
		"""
		Save model config.
		Allow multiple tries to prevent immediate I/O errors.
		"""
		checkpoint = {
			'iteration':iteration,
			'attention':self.attention.state_dict(),
			'attention_init_args':self.attention_init_args,
			'to_predictions_and_loss':self.to_predictions_and_loss.state_dict(),
			'to_predictions_and_loss_init_args':self.to_predictions_and_loss.pack_init_args(),
			'optimizer':self.optimizer.state_dict(),
			'lr_scheduler':self.lr_scheduler.state_dict(),
			'gauss_mix_loss':self.gauss_mix_loss,
			'dirichlet_loss':self.dirichlet_loss,
			'cross_entropy_loss':self.cross_entropy_loss,
			'discrete':self.discrete,
			'quantized_output':self.quantized_output,
			'distributed':self.distributed,
			'random_seed':self.seed,
			'normalize_input_max_L2':dataset.normalize_by_max_L2
		}
		if self.discrete:
			checkpoint['label2ix'] = dataset.label2ix
		torch.save(checkpoint, os.path.join(self.save_dir, 'checkpoint_after-{iteration}-iters.pt'.format(iteration=iteration+1)))
		torch.save(checkpoint, os.path.join(self.save_dir, 'checkpoint.pt'))
		logger.info('Config successfully saved.')


	def retrieve_model(self, checkpoint_path = None, device='cpu', new_locality=None):
		if checkpoint_path is None:
			checkpoint_path = os.path.join(self.save_dir, 'checkpoint.pt')
		self.checkpoint = torch.load(checkpoint_path, map_location='cpu') # Random state needs to be loaded to CPU first even when cuda is available.


		self.attention = model.SelfAttentionEncoder(**self.checkpoint['attention_init_args'])
		attention_state_dict = {('^'+key).replace('^module.', '').replace('^',''):value
							for key,value in self.checkpoint['attention'].items()}
		self.attention.load_state_dict(attention_state_dict)
		if not new_locality is None:
			self.attention.reset_locality(new_locality)
		self.attention_init_args = self.attention.pack_init_args()
		self.bottleneck_layers = self.attention_init_args['bottleneck_layers']
		if self.checkpoint['distributed']:
			self.attention = torch.nn.DataParallel(self.attention)

		self.gauss_mix_loss = False
		self.dirichlet_loss = False
		self.cross_entropy_loss = False
		self.discrete = False
		self.quantized_output = False
		if ('gauss_mix_loss' in self.checkpoint and self.checkpoint['gauss_mix_loss']) or ('log_pdf_loss' in self.checkpoint and self.checkpoint['log_pdf_loss']):
			self.to_predictions_and_loss = model.IsotropicGaussMixLoss(**self.checkpoint['to_predictions_and_loss_init_args'])
			self.gauss_mix_loss = True
		elif 'dirichlet_loss' in self.checkpoint and self.checkpoint['dirichlet_loss']:
			self.to_predictions_and_loss = model.DirichletLoss(**self.checkpoint['to_predictions_and_loss_init_args'])
			self.dirichlet_loss = True
		elif 'cross_entropy_loss' in self.checkpoint and self.checkpoint['cross_entropy_loss']:
			self.to_predictions_and_loss = model.InterProbCrossEntropyLoss(**self.checkpoint['to_predictions_and_loss_init_args'])
			self.cross_entropy_loss = True
		elif 'discrete' in self.checkpoint and self.checkpoint['discrete']:
			self.to_predictions_and_loss = model.DiscreteCrossEntropyLoss(**self.checkpoint['to_predictions_and_loss_init_args'])
			self.discrete = True
			self._label2ix = self.checkpoint['label2ix']
		elif 'quantized_output' in self.checkpoint and self.checkpoint['quantized_output']:
			self.to_predictions_and_loss = model.CrossEntropyLossWithQuantization(**self.checkpoint['to_predictions_and_loss_init_args'])
			self.quantized_output = True
		else:
			self.to_predictions_and_loss = model.MultiL2Loss(**self.checkpoint['to_predictions_and_loss_init_args'])
		self.to_predictions_and_loss.load_state_dict(self.checkpoint['to_predictions_and_loss'])
		if 'normalize_input_max_L2' in self.checkpoint:
			self._normalize_input_max_L2 = self.checkpoint['normalize_input_max_L2']
		else:
			self._normalize_input_max_L2 = None
		self.modules = [self.attention, self.to_predictions_and_loss]
		[m.to(self.device) for m in self.modules]
		self.parameters = lambda:itertools.chain(*[m.parameters() for m in self.modules])

		
		self.seed = self.checkpoint['random_seed']
		return self.checkpoint['iteration']



def get_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('train_data_path', type=str, help='Path to the csv file containing the training data.')
	parser.add_argument("--seq_col", type=str, default='sequence_ix', help="Name of the csv column containing the index of the sequences.")
	parser.add_argument("--time_col", type=str, default='time_ix', help="Name of the csv column containing the discrete time indexes of the sequences.")
	parser.add_argument("--dim_col", type=str, default='dim', help="Name of the csv column containing the index of the dimensions of the data points.")
	parser.add_argument("--val_col", type=str, default='value', help="Name of the csv column containing the data values.")
	parser.add_argument('--individual_col', type=str, default=None, help='Name of the csv column containing the individual info.')
	parser.add_argument('--sep', type=str, default=',', help='The separator of the data')
	parser.add_argument('-S', '--save_root', type=str, default=None, help='Path to the annotationy where results are saved.')
	parser.add_argument('-j', '--job_id', type=str, default='NO_JOB_ID', help='Job ID. For users of computing clusters.')
	parser.add_argument('-s', '--seed', type=int, default=1111, help='random seed')
	parser.add_argument('-d', '--device', type=str, default='cpu', help='Computing device.')
	parser.add_argument('-i', '--iterations', type=int, default=32000, help='# of iterations to train the model.')
	parser.add_argument('-b', '--batch_size', type=int, default=512, help='Batch size for training.')
	parser.add_argument('--attention_hidden_size', type=int, default=512, help='Dimensionality of the hidden space of the attention.')
	parser.add_argument('--num_attention_heads', type=int, default=8, help='# of attention heads.')
	parser.add_argument('--num_attention_layers', type=int, default=1, help='# of layers of attention.')
	parser.add_argument('--bottleneck_layers', type=int, default=[], nargs='+', help='ID# of layers where # of heads is 1 regardless of the value on num_attention_heads. ID# starts with 0.')
	parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
	parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate.')
	parser.add_argument('--decay', type=float, default=0.01, help='Weight decay.')
	parser.add_argument('--betas', type=float, default=[0.9, 0.999], nargs=2, help='Adam coefficients used for computing running averages of gradient and its square.')
	parser.add_argument('--warmup_iters', type=int, default=0, help='# of iterations for warmup.')
	parser.add_argument('--num_workers', type=int, default=1, help='# of workers for dataloading (>=1).')
	parser.add_argument('--saving_interval', type=int, default=200, help='# of iterations in which model parameters are saved once.')
	parser.add_argument('--gauss_mix_loss', action='store_true', help='If selected, the loss is defined by the negative log pdf of an isotropic gaussian mixture.')
	parser.add_argument('--dirichlet_loss', action='store_true', help='If selected, the loss is defined by the negative log pdf of an dirichlet distribution.')
	parser.add_argument('--cross_entropy_loss', action='store_true', help='If selected, the loss is defined by the cross entropy between the target and predicted categorical distributions.')
	parser.add_argument('--discrete', action='store_true', help='Select this option when the data are discrete labels. The model is trained with cross entropy loss.')
	parser.add_argument('--quantized_output', action='store_true', help='Select this option if the continuous data should be quantized in the output. The model is trained with cross entropy loss.')
	parser.add_argument('--quantization_levels', type=int, default=256, help='# of levels of the quantized values.')
	parser.add_argument('--num_predictions', type=int, default=1, help='# of values predicted for each data point.')
	parser.add_argument('--normalize_input_max_L2', action='store_true', help='If selected, normalize the input by the max L2 norm of the data.')
	parser.add_argument('--locality', type=int, default=None, help='If specified, attention is limited to neighbors whose distance threshold is determined by this value.')
	parser.add_argument('--sparsemax', action='store_true', help='If selected, use sparsemax instead of softmax to compute the attention weights in the bottleneck layers.')
	parser.add_argument('--embed_individuals', action='store_true', help='If selected, feed embedding of individual ID to the network.')
	parser.add_argument('--noisy_data', action='store_true', help='If selected, sample data values according to the distribution defined in the train_data_path.')
	parser.add_argument('--prob_col', type=str, default='prob', help='Name of the csv column whose values define the discrete probability of the corresponding row.')
	parser.add_argument('--backward_modeling', action='store_true', help='If selected, language modeling runs backward, from future to past.')

	return parser.parse_args()


def get_save_dir(save_root, job_id_str):
	save_dir = os.path.join(
					save_root,
					job_id_str # + '_START-AT-' + datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S-%f')
				)
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	return save_dir

if __name__ == '__main__':
	args = get_args()

	save_root = args.save_root
	if save_root is None:
		save_root = args.input_root
	save_dir = get_save_dir(save_root, args.job_id)

	individual_json_path = os.path.join(save_dir, 'individual_coding.json')
	if os.path.isfile(individual_json_path):
		with open(individual_json_path, 'r') as f:
			individual2ix = json.load(f)
	else:
		individual2ix = None

	label_json_path = os.path.join(save_dir, 'label_coding.json')
	if os.path.isfile(label_json_path):
		with open(label_json_path, 'r') as f:
			label2ix = json.load(f)
	else:
		label2ix = None

	train_dataset = data_utils.Dataset(
		args.train_data_path,
		seq_col=args.seq_col,
		time_col=args.time_col,
		dim_col=args.dim_col,
		val_col=args.val_col,
		prob_col=args.prob_col,
		normalize_by_max_L2=args.normalize_input_max_L2,
		individual_col=args.individual_col,
		discrete=args.discrete,
		noisy_data=args.noisy_data,
		backward_modeling=args.backward_modeling,
		sep=args.sep,
		individual2ix=individual2ix,
		label2ix=label2ix
	)
	input_size = train_dataset.get_input_size()
	num_individuals = train_dataset.get_num_individuals()
	if individual2ix is None and not num_individuals is None:
		with open(individual_json_path, 'w') as f:
			json.dump(train_dataset.individual2ix, f)

	if label2ix is None:
		with open(label_json_path, 'w') as f:
			json.dump(train_dataset.label2ix, f)
	
	if args.quantized_output:
		step_size = 1.0 / args.quantization_levels
		qs = np.arange(step_size, 1.0, step_size)
		quantization_boundaries = train_dataset.get_quantiles(qs)
	else:
		quantization_boundaries = None

	# Get a model.
	learner = Learner(
				input_size,
				save_dir,
				attention_hidden_size = args.attention_hidden_size,
				num_attention_heads = args.num_attention_heads,
				num_attention_layers = args.num_attention_layers,
				bottleneck_layers=args.bottleneck_layers,
				dropout=args.dropout,
				gauss_mix_loss=args.gauss_mix_loss,
				dirichlet_loss=args.dirichlet_loss,
				cross_entropy_loss=args.cross_entropy_loss,
				num_predictions=args.num_predictions,
				locality=args.locality,
				sparsemax=args.sparsemax,
				num_individuals=num_individuals,
				discrete=args.discrete,
				quantized_output=args.quantized_output,
				quantization_boundaries=quantization_boundaries,
				device = args.device,
				seed = args.seed,
				)

	if args.normalize_input_max_L2:
		logger.info('Inputs are normalized by the max L2 norm of the data.')

	if args.backward_modeling:
		logger.info('Language modeling runs backward, from future to past.')

	if args.noisy_data:
		logger.info('Data are noisy.')
		if learner.retrieval:
			logger.warning('The current implementation does not retrieve the random state of the dataloader workers.')

	assert args.num_workers>0, '--num_workers must be a positive integer.'
	# Train the model.
	learner.learn(
			train_dataset,
			args.iterations,
			args.batch_size,
			learning_rate=args.learning_rate,
			decay = args.decay,
			betas=args.betas,
			warmup_iters=args.warmup_iters,
			saving_interval=args.saving_interval,
			num_workers=args.num_workers
			)