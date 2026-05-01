# Config Explanation

This file explains the parameters in `config/baseline.py` and how the model parameter count is computed.

## Path Parameters

- `data_dir`: Directory containing tokenized `.npy` shards. The training loader expects files with `train` or `val` in the filename, such as `edufineweb_train_000001.npy`.
- `checkpoint_dir`: Directory where model checkpoints are written.
- `log_file`: Training log path. Validation and training loss are appended here.

## Model Parameters

- `block_size`: Maximum sequence length. The contest evaluator expects length up to `1024`, so this should stay `1024`.
- `vocab_size`: GPT-2 tokenizer vocabulary size. The contest expects `50257`, so this should stay `50257`.
- `n_layer`: Number of Transformer blocks.
- `n_head`: Number of attention heads per block.
- `n_embd`: Embedding dimension, also called hidden size or channel size.

For this baseline:

```python
block_size = 1024
vocab_size = 50257
n_layer = 8
n_head = 12
n_embd = 768
```

The head dimension is:

```text
head_dim = n_embd / n_head = 768 / 12 = 64
```

`n_embd` must be divisible by `n_head`.

## Optimization Parameters

- `total_batch_size`: Effective batch size measured in tokens, across all GPUs and gradient accumulation steps.
- `micro_batch_size`: Number of sequences per forward/backward pass on each process.
- `max_steps`: Number of optimizer steps to train.
- `warmup_steps`: Number of initial steps for linear learning-rate warmup.
- `max_lr`: Peak learning rate after warmup.
- `min_lr`: Final learning rate after cosine decay.
- `weight_decay`: AdamW weight decay for matrix-like parameters.
- `beta1`, `beta2`: AdamW momentum coefficients.
- `eps`: AdamW numerical stability epsilon.
- `grad_clip`: Maximum gradient norm.

Gradient accumulation is computed in `train.py` as:

```text
gradient_accumulation_steps = total_batch_size / (micro_batch_size * block_size * world_size)
```

For example, with one GPU:

```text
524288 / (16 * 1024 * 1) = 32 accumulation steps
```

With eight GPUs:

```text
524288 / (16 * 1024 * 8) = 4 accumulation steps
```

## Evaluation and Checkpoint Parameters

- `eval_interval`: Run validation every this many optimizer steps.
- `eval_iters`: Number of validation batches to average.
- `log_interval`: Print and log training loss every this many optimizer steps.
- `checkpoint_interval`: Save a checkpoint every this many optimizer steps.
- `save_step_checkpoints`: If `True`, keep numbered checkpoint files in addition to the latest `checkpoint.pt`.

## Runtime Parameters

- `seed`: Random seed for initialization and data order.
- `compile_model`: Whether to call `torch.compile`.
- `use_amp`: Whether to use automatic mixed precision on CUDA.
- `amp_dtype`: AMP dtype. The baseline uses `bfloat16`.
- `matmul_precision`: PyTorch float32 matmul precision setting.

## Parameter Count Formula

The model uses tied token embedding and output head weights:

```python
self.transformer.wte.weight = self.lm_head.weight
```

Because of this, the token embedding matrix and the `lm_head` matrix are one shared parameter tensor, not two separate tensors.

Let:

```text
V = vocab_size
T = block_size
L = n_layer
C = n_embd
```

The total parameter count is:

```text
token embeddings       = V * C
position embeddings    = T * C
each Transformer block = 12 * C^2 + 13 * C
final LayerNorm        = 2 * C

total = (V * C) + (T * C) + L * (12 * C^2 + 13 * C) + (2 * C)
```

Per Transformer block:

```text
attention qkv weight   = C * 3C
attention qkv bias     = 3C
attention proj weight  = C * C
attention proj bias    = C
MLP fc weight          = C * 4C
MLP fc bias            = 4C
MLP proj weight        = 4C * C
MLP proj bias          = C
LayerNorm 1            = 2C
LayerNorm 2            = 2C

block total            = 12C^2 + 13C
```

## Baseline 96M Count

Current baseline:

```text
V = 50257
T = 1024
L = 8
C = 768
```

Components:

```text
token embeddings       = 50257 * 768 = 38,597,376
position embeddings    = 1024 * 768  =    786,432
each block             = 12 * 768^2 + 13 * 768 = 7,087,872
8 blocks               = 8 * 7,087,872 = 56,702,976
final LayerNorm        = 2 * 768 = 1,536
```

Total:

```text
38,597,376 + 786,432 + 56,702,976 + 1,536 = 96,088,320
```

So the baseline model has:

```text
96,088,320 parameters
```

This is below the 100M contest limit.

## Original GPT-2 Small 124M Count

The original GPT-2-small architecture from `build-nanogpt/train_gpt2.py` uses:

```text
V = 50257
T = 1024
L = 12
C = 768
```

Components:

```text
token embeddings       = 50257 * 768 = 38,597,376
position embeddings    = 1024 * 768  =    786,432
each block             = 7,087,872
12 blocks              = 12 * 7,087,872 = 85,054,464
final LayerNorm        = 1,536
```

Total:

```text
38,597,376 + 786,432 + 85,054,464 + 1,536 = 124,439,808
```

So GPT-2-small is usually referred to as:

```text
124M parameters
```

Note: the local `build-nanogpt/train_gpt2.py` training loop instantiates `GPTConfig(vocab_size=50304)`, padding the vocabulary for hardware efficiency. With `V = 50304`, the same 12-layer model has:

```text
124,475,904 parameters
```

The competition config uses `vocab_size = 50257`, so the relevant original comparison is `124,439,808`.

## What Would Happen Without Tied Embeddings?

If `lm_head` did not share weights with the token embedding table, the model would add another:

```text
V * C = 50257 * 768 = 38,597,376
```

parameters.

That would make the current 8-layer baseline:

```text
96,088,320 + 38,597,376 = 134,685,696
```

which would exceed the 100M limit. Weight tying is therefore important for this baseline.
