"""Training configuration for the nano FineWeb-Edu pipeline."""

# Paths are interpreted relative to the repo root unless absolute.
data_dir = "data/fineweb_edu"
checkpoint_dir = "checkpoints"
checkpoint_filename = "checkpoint.pt"
log_file = "checkpoints/train.log"

# Nano model: RoPE attention, RMSNorm, SwiGLU MLP, tied token/head weights.
#
# Near-100M parameter configuration:
# - 12 layers for more depth
# - n_embd = 640
# - n_head = 10, so head_dim = 64
# - mlp_hidden_dim = 2048
#
# Expected total parameters: about 99M.
block_size = 1024
vocab_size = 50257
n_layer = 12
n_head = 10
n_embd = 640
mlp_hidden_dim = 2048
dropout = 0.0
bias = False

# Keep the same accumulation shape as config/train_gpt2.py:
# 33 * 12 * 1024 = 405,504 tokens per optimizer step.
total_batch_size = 405504
micro_batch_size = 12

# 10,000 steps = about 4.06B tokens.
max_steps = 10000
warmup_steps = 400

max_lr = 3e-4
min_lr = 3e-5
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
eps = 1e-8
grad_clip = 1.0

# Optimizer. On the muon branch, default to Muon for hidden matrix weights
# with AdamW fallback for embeddings, RMSNorm weights, and optional biases.
optimizer = "muon"
muon_lr = 0.02
muon_momentum = 0.95
muon_nesterov = True
muon_ns_steps = 5

# Evaluation and checkpointing.
eval_interval = 250
eval_iters = 50
log_interval = 10
checkpoint_interval = 250
save_step_checkpoints = False

# Runtime.
seed = 1337
compile_model = True
use_amp = True
amp_dtype = "bfloat16"
matmul_precision = "high"

allow_val_train_fallback = False
