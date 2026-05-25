"""Training configuration for the nano FineWeb-Edu pipeline."""

# Paths are interpreted relative to the repo root unless absolute.
data_dir = "data/fineweb_edu"
checkpoint_dir = "checkpoints"
checkpoint_filename = "checkpoint.pt"
log_file = "checkpoints/train.log"

# Nano model: RoPE attention, RMSNorm, SwiGLU MLP, tied token/head weights.
block_size = 1024
vocab_size = 50257
n_layer = 10
n_head = 12
n_embd = 648
dropout = 0.0
bias = False

# Keep the same accumulation shape as config/train_gpt2.py: 33 * 12 * 1024 tokens per step.
total_batch_size = 405504
micro_batch_size = 12
max_steps = 10000
warmup_steps = 400
max_lr = 3e-4
min_lr = 3e-5
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
eps = 1e-8
grad_clip = 1.0

# Evaluation and checkpointing.
eval_interval = 250
eval_iters = 50
log_interval = 10
checkpoint_interval = 500
save_step_checkpoints = False

# Runtime.
seed = 1337
compile_model = True
use_amp = True
amp_dtype = "bfloat16"
matmul_precision = "high"

allow_val_train_fallback = False
