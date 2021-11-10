# coding: utf-8

import torch
import math, collections
from . import sparsemax as sparsemax_module


def pad_flatten_sequence(flatten, batch_sizes, padding_value=0.0, batch_first=False):
	return torch.nn._VF._pad_packed_sequence(
				flatten,
				batch_sizes.cpu(),
				batch_first,
				padding_value,
				batch_sizes.size(0)
				)

def get_mask_on_pad_from_lengths(lengths):
	max_length = lengths.max()
	mask = torch.arange(max_length).view(1,-1).to(lengths.device)>=lengths.view(-1,1)
	return mask

def batch_sizes2lengths(batch_sizes):
	lengths = torch.zeros(batch_sizes[0]).long()
	for bs in batch_sizes:
		lengths[:bs] += 1
	return lengths

def lengths2batch_sizes(lengths):
	batch_sizes = torch.zeros(lengths.max()).long()
	for l in lengths:
		batch_sizes[:l] += 1
	return batch_sizes

def get_locality_mask(seq_length, side_window_length):
	local = torch.zeros((seq_length,)*2)
	side_window_length = min(seq_length-1, side_window_length)
	local += sum(torch.ones(seq_length-abs(dist)).diag(dist)
				for dist in range(-side_window_length,side_window_length+1)
				)
	return (1 - local).bool()

def get_future_mask(seq_length):
	future = torch.ones((seq_length,)*2).triu(diagonal=1).bool()
	return future

class SelfAttentionEncoder(torch.nn.Module):
	def __init__(
			self,
			input_size,
			hidden_size,
			num_heads=8,
			num_layers=1,
			bottleneck_layers=list(),
			dropout=0.0,
			locality=None,
			sparsemax=False,
			num_individuals=None,
			discrete=False,
			):
		super(SelfAttentionEncoder, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_heads = num_heads
		self.num_layers = num_layers
		self.bottleneck_layers = bottleneck_layers
		self.locality = locality
		self.sparsemax = sparsemax
		self.discrete = discrete
		self.dropout = torch.nn.Dropout(dropout)
		if discrete:
			self.to_hidden = torch.nn.Embedding(input_size, hidden_size)
		else:
			self.to_hidden = MLP(input_size, hidden_size, hidden_size)
		layers = [('layer{}'.format(l),
									SelfAttentionLayer(
										hidden_size,
										num_heads=1 if l in self.bottleneck_layers else num_heads,
										dropout=dropout,
										sparsemax=sparsemax if l in self.bottleneck_layers else False,
										bottleneck=l in self.bottleneck_layers
										)
									)
									for l in range(num_layers)
									]
		self.self_attention = torch.nn.Sequential(collections.OrderedDict(layers))
		self.num_individuals = num_individuals
		if num_individuals:
			self.individual_embedding = torch.nn.Embedding(num_individuals, hidden_size)
		pos_encoding_inv_freq = 1 / (10000 ** (torch.arange(0.0, hidden_size, 2.0) / hidden_size))
		self.register_buffer('pos_encoding_inv_freq', pos_encoding_inv_freq)

	def forward(self, input_seqs, lengths=None, batch_sizes=None, individuals=None):
		if isinstance(input_seqs, torch.nn.utils.rnn.PackedSequence):
			batch_sizes = input_seqs.batch_sizes
			input_seqs = input_seqs.data
		assert lengths is not None or batch_sizes is not None, 'lengths or batch_sizes must be specified if input_seqs is not a PackedSequence instance.'
		if lengths is None:
			lengths = batch_sizes2lengths(batch_sizes)
		if batch_sizes is None:
			batch_sizes = lengths2batch_sizes(lengths)
		lengths = lengths.to(input_seqs.device)
		hidden = self.to_hidden(input_seqs)

		if self.num_individuals:
			if isinstance(individuals, torch.nn.utils.rnn.PackedSequence):
				individuals = individuals.data
			individuals = self.individual_embedding(individuals)
		else:
			individuals = torch.zeros_like(hidden)

		hidden = self.dropout(hidden+individuals)

		subbatch_info = self.group_into_subbatches(lengths, batch_sizes)

		pos_encodings, pos_lookup = self.get_pos_encodings(lengths[0])

		input_as_dict = {
			'value':hidden,
			'lengths':lengths,
			'batch_sizes':batch_sizes,
			'pos_encodings':pos_encodings,
			'pos_lookup':pos_lookup,
			'weights':[]
			}
		input_as_dict['subbatch_info'] = subbatch_info
		out_as_dict = self.self_attention(input_as_dict)
		return out_as_dict['value'], out_as_dict['weights']

	def group_into_subbatches(self, lengths, batch_sizes):
		subbatch_size = 0
		subbatch_info = []
		subbatch_lengths = []
		subbatch_ixs = []
		max_length = lengths.max()
		for batch_ix, l in enumerate(lengths):
			if subbatch_lengths and self._is_full_subbatch_size(max_length, subbatch_size+1):
				subbatch_lengths = torch.tensor(subbatch_lengths).to(lengths.device)
				subbatch_sizes = lengths2batch_sizes(subbatch_lengths)
				subbatch_masks = self._get_masks(subbatch_lengths)
				subbatch_token_ixs = self._get_subbatch_token_ixs(batch_sizes, subbatch_ixs, subbatch_lengths)
				subbatch_info.append((subbatch_token_ixs, subbatch_lengths, subbatch_sizes, subbatch_masks))
				max_length = l # Assuming lengths descending.
				subbatch_size = 0
				subbatch_lengths = []
				subbatch_ixs = []
			subbatch_size += 1
			subbatch_ixs.append(batch_ix)
			subbatch_lengths.append(l)
		subbatch_lengths = torch.tensor(subbatch_lengths).to(lengths.device)
		subbatch_sizes = lengths2batch_sizes(subbatch_lengths)
		subbatch_masks = self._get_masks(subbatch_lengths)
		subbatch_token_ixs = self._get_subbatch_token_ixs(batch_sizes, subbatch_ixs, subbatch_lengths)
		subbatch_info.append((subbatch_token_ixs, subbatch_lengths, subbatch_sizes, subbatch_masks))
		return subbatch_info

	def reset_locality(self, locality):
		self.locality = locality

	def _get_masks(self, lengths):
		max_length = lengths.max()
		masks = get_mask_on_pad_from_lengths(lengths)[:,None,None,:]
		masks = masks | get_future_mask(max_length).to(masks.device)[None,None,:,:]
		if not self.locality is None:
			masks = masks | get_locality_mask(max_length, self.locality).to(masks.device)[None,None,:,:]
		return masks

	def _get_subbatch_token_ixs(self, batch_sizes, subbatch_ixs, subbatch_lengths):
		return [bix+cum_bs
				for t,cum_bs in enumerate([0]+batch_sizes.cumsum(0).tolist()[:-1])
				for bix,l_ in zip(subbatch_ixs,subbatch_lengths)
				if t < l_
				]

	def _is_full_subbatch_size(self, max_length, subbatch_size):
		if 512**2*16 < max_length**2*subbatch_size:
			return True
		else:
			return False

	def get_pos_encodings(self, seq_length):
		pos_ixs = torch.arange(seq_length-1,-1,-1.0).to(self.pos_encoding_inv_freq.device)
		pos_lookup = self.transformer_xl_shifter
		sinusoid_input = torch.ger(pos_ixs, self.pos_encoding_inv_freq)
		pos_encodings = torch.cat([sinusoid_input.sin(), sinusoid_input.cos()], dim=-1)
		return pos_encodings, pos_lookup

	def transformer_xl_shifter(self, x, zero_triu=False):
		"""
		Customized version of the RelMultiHeadAttn._rel_shift() in:
		https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py
		x: batch_size x num_heads x query_length x num_pos_types (= memory_length)
		"""
		zero_pad = torch.zeros((*x.size()[:-1], 1),
							device=x.device, dtype=x.dtype)
		x_padded = torch.cat([zero_pad, x], dim=-1)

		x_padded = x_padded.view(*x.size()[:-2], x.size(-1) + 1, x.size(-2))

		x = x_padded[...,1:,:].view_as(x)

		if zero_triu:
			ones = torch.ones((x.size(-2), x.size(-1)))
			x = x * torch.tril(ones, x.size(-1) - x.size(-2))[None,None,:,:]

		return x


	def pack_init_args(self):
		args = {
			'input_size':self.input_size,
			'hidden_size':self.hidden_size,
			'num_heads':self.num_heads,
			'num_layers':self.num_layers,
			'dropout':self.dropout.p,
			'bottleneck_layers':self.bottleneck_layers,
			'locality':self.locality,
			'sparsemax':self.sparsemax,
			'num_individuals':self.num_individuals,
			'discrete':self.discrete,
		}
		return args

class SelfAttentionLayer(torch.nn.Module):
	def __init__(self, hidden_size, num_heads=8, dropout=0.0, bottleneck=False, sparsemax=False):
		super(SelfAttentionLayer, self).__init__()
		hidden_size_per_head = hidden_size // num_heads
		self.to_query = LinearSplit(hidden_size, hidden_size_per_head, num_heads)
		self.to_key = LinearSplit(hidden_size, hidden_size_per_head, num_heads)
		self.to_value = LinearSplit(hidden_size, hidden_size_per_head, num_heads)
		self.transform_pos_encodings = LinearSplit(hidden_size, hidden_size_per_head, num_heads)

		self.attention = DotProductAttention(bottleneck=bottleneck, sparsemax=sparsemax, dropout=dropout)
		self.linear_combine_heads = torch.nn.Linear(hidden_size_per_head*num_heads, hidden_size)
		self.top_feedfoward = MLP(hidden_size, hidden_size, hidden_size, nonlinearity='GELU')
		self.dropout = torch.nn.Dropout(dropout)
		# self.layer_norm = torch.nn.LayerNorm(hidden_size)
		self.layer_norm = BertLayerNorm(hidden_size)

		self.register_parameter(
			'content_bias',
			torch.nn.Parameter(torch.randn(1, num_heads, hidden_size_per_head), requires_grad=True)
			)
		self.register_parameter(
			'pos_bias',
			torch.nn.Parameter(torch.randn(1, num_heads, hidden_size_per_head), requires_grad=True)
			)

	def forward(self, input_as_dict):
		input_ = input_as_dict['value']
		subbatch_info = input_as_dict['subbatch_info']
		pos_encodings = input_as_dict['pos_encodings'] # num_pos_types x hidden_size
		pos_lookup = input_as_dict['pos_lookup']
		query = torch.stack(self.to_query(input_), dim=-2) # sum(lengths) x num_heads x hidden_size_per_head
		key = torch.stack(self.to_key(input_), dim=-2)
		value = torch.stack(self.to_value(input_), dim=-2)
		pos_encodings = torch.stack(self.transform_pos_encodings(pos_encodings), dim=0) # num_heads x num_pos_types x hidden_size_per_head
		query_content_biased = query + self.content_bias
		query_pos_biased = query + self.pos_bias
		attention = []
		weight = []
		max_length = input_as_dict['lengths'].max()
		for subbatch_token_ixs, subbatch_lengths, subbatch_sizes, subbatch_masks in subbatch_info:
			qc,qp,k,v = [pad_flatten_sequence(
						x[subbatch_token_ixs,...],
						subbatch_sizes,
						batch_first=True
					)[0].transpose(1,2).contiguous() # subbatch_size x num_heads x max_lengths x hidden_size_per_head
					for x in [query_content_biased,query_pos_biased, key, value]
					]
			a, w = self.attention(qc, qp, k, v, pos_encodings[...,max_length-k.size(-2):,:], pos_lookup, mask=subbatch_masks)
			a = a.transpose(1,2).contiguous().view(a.size(0), a.size(2), -1)
			a,w = zip(*[(a_per_seq[:l,...],w_per_seq[:,:l,:l]) for a_per_seq,w_per_seq,l in zip(a, w, subbatch_lengths)])
			attention += list(a)
			weight += list(w)
		attention = torch.nn.utils.rnn.pack_sequence(attention).data
		attention = self.linear_combine_heads(attention)
		attention = self.dropout(attention)
		attention = self.layer_norm(attention + input_) # Attention should encode the DISTANCE between the target and the attended time steps. So, no need to add the positional encodings.

		out = self.top_feedfoward(attention)
		out = self.dropout(out)
		out = self.layer_norm(attention + out)
		input_as_dict['value'] = out
		input_as_dict['weights'].append(weight)
		return input_as_dict


class BertLayerNorm(torch.nn.Module):
	def __init__(self, hidden_size, eps=1e-12):
		"""Construct a layernorm module in the TF style (epsilon inside the square root).
		Copied from pytorch-transformers project:
		https://github.com/huggingface/pytorch-transformers/blob/master/pytorch_transformers/modeling_bert.py
		"""
		super(BertLayerNorm, self).__init__()
		self.weight = torch.nn.Parameter(torch.ones(hidden_size))
		self.bias = torch.nn.Parameter(torch.zeros(hidden_size))
		self.variance_epsilon = eps

	def forward(self, x):
		u = x.mean(-1, keepdim=True)
		s = (x - u).pow(2).mean(-1, keepdim=True)
		x = (x - u) / torch.sqrt(s + self.variance_epsilon)
		return self.weight * x + self.bias


class LinearSplit(torch.nn.Module):
	def __init__(self, input_size, output_size, num_splits):
		super(LinearSplit, self).__init__()
		self.linears = torch.nn.ModuleList([
								torch.nn.Linear(input_size, output_size)
								for ix in range(num_splits)
							])
		
	def forward(self, x):
		return [l(x) for l in self.linears]

class DotProductAttention(torch.nn.Module):
	def __init__(self, bottleneck=False, sparsemax=False, dropout=0.0):
		super(DotProductAttention, self).__init__()
		self.bottleneck = bottleneck
		if sparsemax:
			self.softmax = sparsemax_module.Sparsemax(dim=-1)
		else:
			self.softmax = torch.nn.Softmax(dim=-1)
		self.dropout = torch.nn.Dropout(dropout)

	def forward(self, query_content_biased, query_pos_biased, key, value, pos_encodings, pos_lookup, mask=None):
		"""
		Version adopted in the Transformer XL (Dai et al., 2019).
		https://arxiv.org/abs/1901.02860

		query: batch_size x num_heads x length_1 x hidden_size
		key, value: batch_size x num_heads x length_2 x hidden_size
		pos_encodings: num_heads x num_pos_types x hidden_size
		content_bias, pos_bias: num_heads x hidden_size
		pos_lookup: function that shifts the positional encodings in an appropriate way.

		TODOs: Convert query and weight to sparse tensors (to save CUDA memory) as soon as
		torch.sparse.matmul() gets supported.
		"""
		weight = torch.einsum(
					'bnid,bnjd->bnij',
					(
						query_content_biased,
						key
					)
					)
		weight = weight + pos_lookup(
					torch.einsum(
						'bnid,npd->bnip',
						(
							query_pos_biased,
							pos_encodings
						)
						)
					)
		weight = weight / math.sqrt(key.size(-1))
		if mask is None:
			weight = self.softmax(weight)
		else:
			weight = weight.masked_fill(mask,torch.finfo().min)
			weight = self.softmax(weight)
			weight = weight.masked_fill(mask,0.0)
		weight = self.dropout(weight)

		out = torch.einsum('bnij,bnjd->bnid', (weight, value))
		return out, weight

	# def get_random_one_hot_weight(self, weight_size):
	# 	weight = torch.zeros((weight_size[0],weight_size[-1])).float()
	# 	weight[...,0] = 1.0
	# 	weight = torch.stack([
	# 				w[torch.randint(l, weight_size[1:])]
	# 				for w,l in zip(weight,memory_length)],
	# 				dim=0)
	# 	return weight




class MLP(torch.jit.ScriptModule):
# class MLP(torch.nn.Module):
	"""
	Multi-Layer Perceptron.
	"""
	def __init__(self, input_size, hidden_size, output_size, nonlinearity='GELU'):
		super(MLP, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.nonlinearity = nonlinearity
		if nonlinearity=='GELU':
			nonlinearity = GELU()
		else:
			nonlinearity = getattr(torch.nn, nonlinearity)()
		self.whole_network = torch.nn.Sequential(
			torch.nn.Linear(input_size, hidden_size),
			nonlinearity,
			torch.nn.Linear(hidden_size, output_size)
			)

	@torch.jit.script_method
	def forward(self, batched_input):
		return self.whole_network(batched_input)

	def pack_init_args(self):
		init_args = {
			'input_size':self.input_size,
			'hidden_size':self.hidden_size,
			'output_size':self.output_size,
			'nonlinearity':self.nonlinearity
		}
		return init_args


class GELU(torch.jit.ScriptModule):
	"""
	Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
	# Copied from BERT-pytorch:
	# https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/model/utils/gelu.py
	UPDATED: Copied from pytorch-transformers. The previous version is of OpenAI GPT.
	https://github.com/huggingface/pytorch-transformers/blob/master/pytorch_transformers/modeling_bert.py
	"""
	# __constants__ = ['sqrt_2_over_pi']
	__constants__ = ['sqrt_2']
	def __init__(self):
		super(GELU, self).__init__()
		# self.sqrt_2_over_pi = math.sqrt(2 / math.pi)
		self.sqrt_2 = math.sqrt(2)

	@torch.jit.script_method
	def forward(self, x):
		# return 0.5 * x * (1 + torch.tanh(self.sqrt_2_over_pi * (x + 0.044715 * torch.pow(x, 3))))
		return x * 0.5 * (1.0 + torch.erf(x / self.sqrt_2))



class MultiL2Loss(torch.nn.Module):
	def __init__(self, in_features, out_features, num_predictions):
		super(MultiL2Loss, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.num_predictions = num_predictions
		self.to_weights = torch.nn.Linear(in_features, num_predictions)
		self.to_predictions = torch.nn.ModuleList([
								torch.nn.Linear(in_features, out_features)
								for ix in range(num_predictions)]
								)
		self.mse_loss = torch.nn.MSELoss(reduction='none')

	def forward(self, x, target, mask=None):
		weights = self.to_weights(x).softmax(-1)
		predictions = torch.stack([m(x) for m in self.to_predictions], dim=1)
		if not mask is None:
			predictions = predictions * mask[:,None,:]
			target = target * mask
		loss = self.mse_loss(predictions, target[:,None,:].expand(-1,predictions.size(1),-1))
		loss = (loss*weights[:,:,None]).sum((1,2))
		return loss

	def pack_init_args(self):
		args = {
			'in_features':self.in_features,
			'out_features':self.out_features,
			'num_predictions':self.num_predictions,
			}
		return args



class IsotropicGaussMixLoss(torch.nn.Module):
	def __init__(self, in_features, out_features, num_predictions):
		super(IsotropicGaussMixLoss, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.num_predictions = num_predictions
		self.to_weights = torch.nn.Linear(in_features, num_predictions)
		self.to_mean = torch.nn.ModuleList([
								torch.nn.Linear(in_features, out_features)
								for ix in range(num_predictions)]
								)
		self.to_log_var = torch.nn.ModuleList([
								torch.nn.Linear(in_features, out_features)
								for ix in range(num_predictions)]
								)

	def forward(self, x, target, mask=None):
		log_weights, mean, log_var = self.get_parameters(x)
		log_prob = self.log_pdf_isotropic_mix(target, mean, log_var, log_weights)
		if not mask is None:
			log_prob = log_prob*mask.view(-1) # Remove the dummy dimension.
		return -log_prob

	def get_parameters(self, x):
		log_weights = self.to_weights(x).log_softmax(-1)
		mean, log_var = zip(*[(m(x),v(x)) for m,v in zip(self.to_mean, self.to_log_var)])
		return log_weights, torch.stack(mean, dim=1), torch.stack(log_var, dim=1)


	def log_pdf_isotropic_mix(self, value, mean, log_variance, log_weights):
		log_prob = self.log_pdf_isotropic_gaussian(value[:,None,:], mean, log_variance)
		log_prob = (log_prob + log_weights).logsumexp(-1)
		return log_prob

	@staticmethod
	def log_pdf_isotropic_gaussian(value, mean, log_variance):
		value_mean_diff = value - mean
		return - 0.5 * (
					math.log(2 * math.pi)
					+ log_variance
					+ value_mean_diff * (-log_variance).exp() * value_mean_diff
					).sum(-1)

	def pack_init_args(self):
		args = {
			'in_features':self.in_features,
			'out_features':self.out_features,
			'num_predictions':self.num_predictions,
			}
		return args


class DirichletLoss(torch.nn.Module):
	def __init__(self, in_features, num_classes, base_count=1.0):
		super(DirichletLoss, self).__init__()
		self.in_features = in_features
		self.num_classes = num_classes
		self.to_shape = torch.nn.Linear(in_features, num_classes)
		self.to_scale = torch.nn.Linear(in_features, 1)
		# self.relu = torch.nn.ReLU()
		# self.selu = torch.nn.SELU()
		# self.min_non_linearity = -1.6732632423543772848170429916717*1.0507009873554804934193349852946
		self.base_count = base_count
		

	def forward(self, x, target, mask=None):
		shape,scale = self.get_parameters(x)
		dirichlet = torch.distributions.dirichlet.Dirichlet(shape*scale+self.base_count)
		log_prob = dirichlet.log_prob(target+torch.finfo().tiny)
		if not mask is None:
			log_prob = log_prob*mask.view(-1) # Remove the dummy dimension.
		return -log_prob

	def get_parameters(self, x):
		shape = self.to_shape(x).softmax(-1)
		scale = self.to_scale(x)
		# scale = self.selu(scale)-self.min_non_linearity
		scale = scale.exp()
		return shape,scale


	def pack_init_args(self):
		args = {
			'in_features':self.in_features,
			'num_classes':self.num_classes,
			'base_count':self.base_count
			}
		return args


class InterProbCrossEntropyLoss(torch.nn.Module):
	def __init__(self, in_features, num_classes):
		super(InterProbCrossEntropyLoss, self).__init__()
		self.in_features = in_features
		self.num_classes = num_classes
		self.fc = torch.nn.Linear(in_features, num_classes)
		

	def forward(self, x, target, mask=None):
		log_prob = self.fc(x).log_softmax(-1)
		loss = -(target * log_prob).sum(-1)
		if not mask is None:
			loss = loss*mask.view(-1) # Remove the dummy dimension.
		return loss

	def pack_init_args(self):
		args = {
			'in_features':self.in_features,
			'num_classes':self.num_classes,
			}
		return args

class DiscreteCrossEntropyLoss(torch.nn.Module):
	def __init__(self, in_features, num_classes):
		super(DiscreteCrossEntropyLoss, self).__init__()
		self.in_features = in_features
		self.num_classes = num_classes
		self.fc = torch.nn.Linear(in_features, num_classes)
		self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')

	def forward(self, x, target, mask=None):
		x = self.fc(x)
		loss = self.cross_entropy_loss(x, target)
		if not mask is None:
			loss = loss*mask
		return loss

	def pack_init_args(self):
		args = {
			'in_features':self.in_features,
			'num_classes':self.num_classes,
			}
		return args


class CrossEntropyLossWithQuantization(torch.nn.Module):
	def __init__(self, in_features, boundaries):
		"""
		boundaries (torch.Tensor; ndim x levels-1): Boundaries for the quantization. The level dim should be sorted.
		"""
		super(CrossEntropyLossWithQuantization, self).__init__()
		self.in_features = in_features
		self.boundaries = boundaries
		self.fc = torch.nn.Linear(in_features, sum(self.boundaries.size()))
		self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')

	def forward(self, x, target, mask=None):
		batch_size,ndim = target.size()
		x = self.fc(x)
		x = x.view(-1,self.boundaries.size(-1))
		target = self._quantize(target)
		loss = self.cross_entropy_loss(x, target)
		loss = loss.view(batch_size,-1).mean(-1)
		if not mask is None:
			loss = loss*mask
		return loss

	def _quantize(self, target):
		target = (target[:,:,None]>=self.boundaries[None,:,:]).sum(-1).view(-1)
		return target

	def pack_init_args(self):
		args = {
			'in_features':self.in_features,
			'boundaries':self.boundaries,
			}
		return args