"""Training configuration for the nano blended-data pipeline."""

# Paths are interpreted relative to the repo root unless absolute.
data_dir = "data/blend/fineweb_edu"
val_data_path = "val.bin"
checkpoint_dir = "checkpoints"
checkpoint_filename = "checkpoint.pt"
log_file = "checkpoints/train.log"

# Match the TA-style validation blend during training:
# FineWeb-Edu 50%, Wikipedia 20%, papers 15%, books 15%.
# With grad_accum_steps=33, each optimizer step consumes:
# 16 FineWeb-Edu, 7 Wikipedia, 5 Papers, 5 Books microbatches.
# Papers are sampled as 70% arXiv article + 30% PubMed article over time.
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
            {"name": "arxiv", "data_dir": "data/blend/papers_arxiv", "weight": 0.7},
            {"name": "pubmed", "data_dir": "data/blend/papers_pubmed", "weight": 0.3},
        ],
    },
    {
        "name": "books",
        "data_dir": "data/blend/books",
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

# Architecture ablations.
use_qk_norm = True
qk_norm_scale_init = None  # None means sqrt(head_dim)
zero_init_residual_projections = False

# Sequence-length curriculum. The model keeps block_size=1024 as its maximum
# context, while training/eval batches use the active runtime sequence length.
use_sequence_curriculum = True
sequence_curriculum = [
    (0, 512),
    (2000, 1024),
]

# Keep optimizer-step tokens fixed as sequence length changes:
# 66 * 12 * 512 = 33 * 12 * 1024 = 405,504 tokens per optimizer step.
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
muon_lr = 0.015
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
