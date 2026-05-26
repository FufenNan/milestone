# Config Explanation

This file explains the main parameters in `config/config.py`.

## Paths

- `data_dir`: Directory containing tokenized `.bin` shards such as `fineweb_train_000001.bin`.
- `checkpoint_dir`: Directory where checkpoints are written.
- `checkpoint_filename`: Latest checkpoint name. The training path uses `checkpoint.pt`.
- `log_file`: Training log path.

## Model

Current v2 nano config:

```python
block_size = 1024
vocab_size = 50257
n_layer = 12
n_head = 10
n_embd = 640
mlp_hidden_dim = 2048
dropout = 0.0
bias = False
```

The model uses token embeddings, RoPE attention, RMSNorm, SwiGLU MLPs, and a tied output head. It has:

```text
99,027,200 parameters
```

This stays below the 100M limit.

## Optimization

- `total_batch_size`: Effective token batch size across all GPUs and accumulation steps.
- `micro_batch_size`: Number of sequences per forward/backward pass per process.
- `max_steps`: Number of optimizer steps.
- `warmup_steps`: Linear warmup length.
- `max_lr` / `min_lr`: Cosine schedule range.
- `optimizer`: Optimizer choice. The `muon` branch defaults to `"muon"`.
- `weight_decay`, `beta1`, `beta2`, `eps`: AdamW settings for the baseline optimizer and Muon fallback groups.
- `muon_lr`, `muon_momentum`, `muon_nesterov`, `muon_ns_steps`: Muon settings for hidden matrix weights.
- `grad_clip`: Maximum gradient norm.

Gradient accumulation is computed in `train.py` as:

```text
total_batch_size / (micro_batch_size * block_size * world_size)
```

For the default single-process config:

```text
405504 / (12 * 1024 * 1) = 33 accumulation steps
```

## Evaluation And Checkpointing

- `eval_interval`: Run validation every this many optimizer steps.
- `eval_iters`: Number of validation batches to average.
- `log_interval`: Print and log training loss every this many steps.
- `checkpoint_interval`: Save `checkpoint.pt` every this many steps.
- `save_step_checkpoints`: Also save numbered checkpoints when enabled.

If `allow_val_train_fallback = True` and no validation shard exists, local smoke tests can use train shards for validation.

## Runtime

- `seed`: Random seed.
- `compile_model`: Whether to call `torch.compile`.
- `use_amp`: Whether to use automatic mixed precision on CUDA.
- `amp_dtype`: AMP dtype.
- `matmul_precision`: PyTorch float32 matmul precision setting.
