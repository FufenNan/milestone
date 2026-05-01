# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
import time

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# this makes total number of tokens be 300B
# max_iters = 600000
# lr_decay_iters = 600000
max_iters = 1500
lr_decay_iters = 1500

# eval stuff
eval_interval = 100
eval_iters = 50
log_interval = 10

# weight decay
weight_decay = 1e-1

# dataset
dataset = 'fineweb10B'

# model
model_name = 'abt2'
# model_name = 'nano'
n_layer = 10
n_head = 12
n_embd = 720

# model_name = 'gpt2'
# n_layer = 10
# n_head = 12 #768
# n_embd = 720

out_dir = f"{model_name}_{time.strftime('%m%d_%H%M%S')}"

# wandb
wandb_log = True
wandb_project = 'milestone_large'
wandb_run_name = f"{model_name}_{time.strftime('%m%d_%H%M%S')}"

#python train.py config/train_gpt2.py