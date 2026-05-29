"""Training configuration for the nano blended-data pipeline."""

# Paths are interpreted relative to the repo root unless absolute.
data_dir = "data/blend/fineweb_edu"
val_data_path = "val.bin"
checkpoint_dir = "checkpoints"
checkpoint_filename = "checkpoint.pt"
log_file = "checkpoints/train.log"

# Match the TA-style validation blend during training, using the new
# no-PubMed PG19 blend from Drive dataset/blend_new:
# FineWeb-Edu 50%, Wikipedia 20%, arXiv papers 15%, PG19 15%.
# With grad_accum_steps=33, each optimizer step consumes:
# 16 FineWeb-Edu, 7 Wikipedia, 5 arXiv Papers, 5 PG19 microbatches.
train_data_mix = [
    {
        "name": "fineweb_edu",
        "data_dir": "data/blend/fineweb_edu",
        "micro_batches": 16,
    },
    {
        "name": "wikipedia",
        "data_dir": "data/blend/wikipedia",
        "micro_batches": 7,
    },
    {
        "name": "papers",
        "micro_batches": 5,
        "subsets": [
            {"name": "arxiv", "data_dir": "data/blend/papers_arxiv", "weight": 1.0},
        ],
    },
    {
        "name": "pg19",
        "data_dir": "data/blend/pg19",
        "micro_batches": 5,
    },
]

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

# Use a 25,000-step LR schedule so multiple 10,000-step sessions continue on
# the same cosine curve. Each notebook training session passes --steps 10000.
max_steps = 25000
steps_this_run = 10000
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
