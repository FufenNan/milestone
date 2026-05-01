import importlib.util
import inspect
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 8
    n_head: int = 12
    n_embd: int = 768


def _load_config_from_file():
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "config", "baseline.py"),
        os.path.join(os.path.dirname(here), "config", "baseline.py"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        spec = importlib.util.spec_from_file_location("baseline_config", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        values = {}
        for key in GPTConfig.__dataclass_fields__:
            if hasattr(module, key):
                values[key] = getattr(module, key)
        return GPTConfig(**values)
    return GPTConfig()


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        if T > self.config.block_size:
            raise ValueError(f"Cannot forward sequence length {T}; block size is {self.config.block_size}")
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, eps, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = {"fused": use_fused} if fused_available else {}
        return torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            **extra_args,
        )


def _clean_state_dict(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod.") :]
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    return cleaned


def _config_from_checkpoint(value):
    if isinstance(value, GPTConfig):
        return value
    if isinstance(value, dict):
        return GPTConfig(**{k: value[k] for k in GPTConfig.__dataclass_fields__ if k in value})
    values = {}
    for key in GPTConfig.__dataclass_fields__:
        if hasattr(value, key):
            values[key] = getattr(value, key)
    return GPTConfig(**values) if values else None


def _torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model(checkpoint_path: str, device: str = "cuda") -> torch.nn.Module:
    checkpoint = _torch_load(checkpoint_path, device)
    config = _load_config_from_file()

    if isinstance(checkpoint, dict) and "model_config" in checkpoint:
        config = _config_from_checkpoint(checkpoint["model_config"])
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "config" in checkpoint and "model" in checkpoint:
        config = _config_from_checkpoint(checkpoint["config"]) or config
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        if "config" in checkpoint:
            config = _config_from_checkpoint(checkpoint["config"]) or config
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model = GPT(config)
    model.load_state_dict(_clean_state_dict(state_dict))
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    config = _load_config_from_file()
    model = GPT(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    assert n_params <= 100_000_000
    x = torch.randint(0, config.vocab_size, (1, min(16, config.block_size)))
    logits = model(x)
    print(f"Output shape: {tuple(logits.shape)}")
