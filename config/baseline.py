"""Baseline training configuration for the <=100M GPT model."""

# Paths are interpreted relative to the gpt100m repo root unless absolute.
data_dir = "data/edu_fineweb10B"
checkpoint_dir = "checkpoints"
log_file = "checkpoints/train.log"

# Model. This is the build-nanogpt GPT-2 architecture scaled below 100M params.
block_size = 1024
vocab_size = 50257
n_layer = 8
n_head = 12
n_embd = 768

# Optimization.
total_batch_size = 524288
micro_batch_size = 16
max_steps = 19073
warmup_steps = 715
max_lr = 6e-4
min_lr = 6e-5
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
eps = 1e-8
grad_clip = 1.0

# Evaluation and checkpointing.
eval_interval = 250
eval_iters = 20
log_interval = 1
checkpoint_interval = 5000
save_step_checkpoints = True

# Runtime.
seed = 1337
compile_model = False
use_amp = True
amp_dtype = "bfloat16"
matmul_precision = "high"
