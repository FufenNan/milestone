import time

out_dir = f"nano_fineweb_edu_{time.strftime('%m%d_%H%M%S')}"
eval_interval = 250
eval_iters = 50
log_interval = 10

always_save_checkpoint = False

wandb_log = True
wandb_project = '251B'
wandb_group_name = 'fineweb_edu'
wandb_run_name = out_dir

dataset = 'fineweb_edu'
gradient_accumulation_steps = 33
batch_size = 12
block_size = 1024

model_name = 'nano'
n_layer = 10
n_head = 10
n_embd = 640
dropout = 0.0

# adamw optimizer
learning_rate = 3e-4
max_iters = 5000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay settings
decay_lr = True
warmup_iters = 200
lr_decay_iters = 5000
min_lr = 3e-5