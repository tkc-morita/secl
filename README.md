# Context dependency estimation based on Transformer language modeling
This repository implements the estimation of context dependency in birdsong based on Transformer language modeling, explored in [Morita et al. (2020)](https://doi.org/10.1101/2020.05.09.083907).

## Data preparation

Data need to be formatted in a csv file that at least containing:

| sequence_id | time_id | value | individual |
| ---         | ---     | ---   | ---        |
| 0           | 0       | a     | B01        |
| 0           | 1       | b     | B01        |
| 0           | 2       | c     | B01        |
| ...         | ...     | ...   | ...        |
| 1           | 0       | d     | B02        |
| 1           | 1       | e     | B02        |
| ...         | ...     | ...   | ...        |

where 
- sequence_id indexes sequences.
- time_id indexes time steps per sequence.
- value encodes the token value (e.g., word/syllable category).
- individual (optional) encodes the individual that produced the corresponding data. (Note that individual is not used to distinguish sequences, so you need to assign unique sequence_id's across different individuals.)

The specific name of the columns can be changed (e.g. "bout_id" for sequence_id).
In the below, we refer to the column names by shell variables as follow:

```sh
seq_col=sequence_id
time_col=time_id
val_col=value
individual_col=individual
```

## Training

First train the Transformer language model.

```sh
python learning.py $train_data --seq_col $seq_col --time_col $time_col --val_col $val_col --individual_col $individual_col -S $save_root -j $BOTTOM_DIRECTORY_NAME \
-d cuda -b 128 --warmup_iters 1000 --saving_interval 500 -i 20000 --learning_rate 0.001 --attention_hidden_size 512 --num_attention_layers 6 --num_attention_heads 8 
--discrete --embed_individuals 
```

- The first line (before the backslash `\`) specifies the data path and column names etc.
  - Your result will be saved in `$save_root/$BOTTOM_DIRECTORY_NAME` (set anything for each of the variables).
- The second line lists the values on free parameters used in [Morita et al. (2020)](https://doi.org/10.1101/2020.05.09.083907).
  - `-d cuda` assumes that a NVIDIA graphic card is available. [Morita et al. (2020)](https://doi.org/10.1101/2020.05.09.083907) used a single RTX 2080Ti (11GB VRAM).
  - `--discrete` is necessary for modeling time series data of discrete symbols; otherwise, the program assumes a seires of real-valued vectors, which is not supported.
  - Without `--embed_individuals`, the program ignores `$individual_col` and builds a non-conditional language model without referring to the background info. about speakers.
  - See `python learning.py --help` for info. about other options.


## Predict test data based on full and truncated contexts

After training, predict test data using the trained model.

You need to run `loss_evaluation.py` multiple times to compute the prediction performance based on
- full context, and
- truncated context of length 1, 2, ..., L for some arbitrary L.

To get full-context predictions,
```sh
save_dir=$save_root/$BOTTOM_DIRECTORY_NAME
python loss_evaluation.py $save_dir/checkpoint.pt $test_data --seq_col $seq_col --time_col $time_col --val_col $val_col --individual_col $individual_col -S $save_dir/loss/full_context.csv -d cuda -b 256 
```
The results are saved in `$save_root/$BOTTOM_DIRECTORY_NAME/loss/full_context.csv`.

Truncated-context predictions can be computed by providing an additional option `--locality`.
The following computes the predictions for each truncated context of length 1 to 15.
```sh
for locality in `seq 1 15`; do
	python loss_evaluation.py $save_dir/checkpoint.pt $test_data --seq_col $seq_col --time_col $time_col --val_col $val_col --individual_col $individual_col -S $save_dir/loss/${locality}-local_context.csv -d cuda -b 256 --locality $locality
done
```


## Compute the statistically effective context length (SECL)

Finally, run `compute_SECL.py` to compute the statistically effective context length (SECL), which is the bootstrapped version of effective context length (ECL) proposed by [Khandelwal et al. (2018)](http://dx.doi.org/10.18653/v1/P18-1027).

```sh
loss_dir=$save_dir/loss
python misc/compute_SECL.py -S $loss_dir/SECL.csv -f $loss_dir/full_context.csv -l 1 $loss_dir/1-local_context.csv 2 $loss_dir/2-local_context.csv 3 $loss_dir/3-local_context.csv 4 $loss_dir/4-local_context.csv 5 $loss_dir/5-local_context.csv 6 $loss_dir/6-local_context.csv 7 $loss_dir/7-local_context.csv 8 $loss_dir/8-local_context.csv 9 $loss_dir/9-local_context.csv 10 $loss_dir/10-local_context.csv 11 $loss_dir/11-local_context.csv 12 $loss_dir/12-local_context.csv 13 $loss_dir/13-local_context.csv 14 $loss_dir/14-local_context.csv 15 $loss_dir/15-local_context.csv
```

- `-f` option specifies the path to the full-context results.
- `-l` option lists pairs of truncated context length and the path to the results (e.g. `-l 1 path1 2 path2 ...`).