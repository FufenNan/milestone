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
    n_layer: int = 10
    n_head: int = 12
    n_embd: int = 648
    dropout: float = 0.0
    bias: bool = False


def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(q, k, freqs_cis):
    B, T, H, D = q.shape
    assert D % 2 == 0, "RoPE requires an even head dimension"
    q_dtype = q.dtype
    k_dtype = k.dtype
    q = q.float().view(B, T, H, D // 2, 2)
    k = k.float().view(B, T, H, D // 2, 2)
    q_complex = torch.view_as_complex(q)
    k_complex = torch.view_as_complex(k)
    freqs_cis = freqs_cis[:T].unsqueeze(0).unsqueeze(2)
    q_out = torch.view_as_real(q_complex * freqs_cis).flatten(-2)
    k_out = torch.view_as_real(k_complex * freqs_cis).flatten(-2)
    return q_out.to(q_dtype), k_out.to(k_dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(self.head_dim, config.block_size),
            persistent=False,
        )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, self.freqs_cis)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * config.n_embd
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.w2 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.proj.NANOGPT_SCALE_INIT = 1
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.proj(F.silu(self.w1(x)) * self.w2(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=RMSNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= self.config.n_layer**-0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        _, T = idx.size()
        if T > self.config.block_size:
            raise ValueError(f"Cannot forward sequence length {T}; block size is {self.config.block_size}")

        x = self.transformer.drop(self.transformer.wte(idx))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
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

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def _load_config_from_file():
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "config", "config.py"),
        os.path.join(here, "config", "baseline.py"),
        os.path.join(os.path.dirname(here), "config", "config.py"),
        os.path.join(os.path.dirname(here), "config", "baseline.py"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        spec = importlib.util.spec_from_file_location("train_config", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        values = {}
        for key in GPTConfig.__dataclass_fields__:
            if hasattr(module, key):
                values[key] = getattr(module, key)
        return GPTConfig(**values)
    return GPTConfig()


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
        config = _config_from_checkpoint(checkpoint["model_config"]) or config
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "config" in checkpoint and "model" in checkpoint:
        config = _config_from_checkpoint(checkpoint["config"]) or config
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        config = _config_from_checkpoint(checkpoint.get("config", None)) or config
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
