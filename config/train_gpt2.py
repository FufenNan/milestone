# config for milestone-scale pretraining on FineWeb-Edu.
# Intended for a single Colab A100 80GB run:
# $ python train.py config/train_gpt2.py
import time

# these make the total batch size be ~0.4M tokens
# 12 batch size * 1024 block size * 33 gradaccum = 405,504
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 33

# this makes total number of tokens be ~2.03B
# 12 * 1024 * 33 * 5000 = 2,027,520,000
max_iters = 5000
lr_decay_iters = 5000

# eval stuff
eval_interval = 100
eval_iters = 50
log_interval = 10

# weight decay
weight_decay = 1e-1
learning_rate = 3e-4
min_lr = 3e-5
warmup_iters = 200

# dataset
dataset = 'fineweb_edu'

# model
model_name = 'abt2'
# total params are ~98.4M with vocab_size=50304 and block_size=1024
n_layer = 10
n_head = 10
n_embd = 640

# model_name = 'gpt2'
# n_layer = 9
# n_head = 12
# n_embd = 672

out_dir = f"{model_name}_{time.strftime('%m%d_%H%M%S')}"

# wandb
wandb_log = True
wandb_project = 'milestone_large'
wandb_run_name = f"{model_name}_{time.strftime('%m%d_%H%M%S')}"

# python train.py config/train_gpt2.py
