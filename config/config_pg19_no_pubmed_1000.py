"""1000-step ablation config using PG19 instead of books and no PubMed."""

# Paths are interpreted relative to the repo root unless absolute.
data_dir = "data/blend/fineweb_edu"
val_data_path = "val.bin"
checkpoint_dir = "checkpoints"
checkpoint_filename = "checkpoint.pt"
log_file = "checkpoints/train.log"

# PG19 ablation blend:
# Keep the current microbatch proportions, but replace books with PG19 and
# remove PubMed from the papers source.
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
block_size = 1024
vocab_size = 50257
n_layer = 12
n_head = 10
n_embd = 640
mlp_hidden_dim = 2048
dropout = 0.0
bias = False

# 33 * 12 * 1024 = 405,504 tokens per optimizer step.
total_batch_size = 405504
micro_batch_size = 12

# 1,000 steps = about 405M tokens.
max_steps = 1000
warmup_steps = 40

max_lr = 3e-4
min_lr = 3e-5
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
eps = 1e-8
grad_clip = 1.0

optimizer = "muon"
muon_lr = 0.02
muon_momentum = 0.95
muon_nesterov = True
muon_ns_steps = 5

# Slightly larger validation average for the shorter ablation runs.
eval_interval = 100
eval_iters = 100
log_interval = 10
checkpoint_interval = 250
save_step_checkpoints = False

seed = 1337
compile_model = True
use_amp = True
amp_dtype = "bfloat16"
matmul_precision = "high"

allow_val_train_fallback = False
