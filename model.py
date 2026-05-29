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
    n_layer: int = 12
    n_head: int = 10
    n_embd: int = 640
    mlp_hidden_dim: int = 2048
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


def zeropower_via_newtonschulz5(g, steps=5, eps=1e-7):
    if g.ndim != 2:
        raise ValueError("Muon only supports 2D parameters")

    dtype = g.dtype
    x = g.float()
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.t()

    x = x / (x.norm() + eps)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        xx_t = x @ x.t()
        x = a * x + (b * xx_t + c * xx_t @ xx_t) @ x

    if transposed:
        x = x.t()
    return x.to(dtype=dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0.0, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError("Muon only supports 2D parameters")

                grad = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)
                update = grad.add(buf, alpha=momentum) if nesterov else buf
                update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                update.mul_(max(1.0, p.size(0) / p.size(1)) ** 0.5)

                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)
                p.add_(update, alpha=-lr)

        return loss


class CombinedOptimizer:
    def __init__(self, optimizers):
        self.optimizers = [opt for opt in optimizers if opt is not None]
        self.param_groups = []
        for opt in self.optimizers:
            self.param_groups.extend(opt.param_groups)

    def zero_grad(self, set_to_none=True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        if closure is not None and len(self.optimizers) != 1:
            raise RuntimeError("CombinedOptimizer does not support closures with multiple optimizers")
        loss = None
        for opt in self.optimizers:
            loss = opt.step(closure=closure) if closure is not None else opt.step()
        return loss

    def state_dict(self):
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict):
        for opt, opt_state in zip(self.optimizers, state_dict["optimizers"]):
            opt.load_state_dict(opt_state)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        assert self.head_dim % 2 == 0, "RoPE requires an even head dimension"
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
        hidden_dim = config.mlp_hidden_dim
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
        assert config.n_embd % config.n_head == 0
        assert (config.n_embd // config.n_head) % 2 == 0
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
                std *= (2 * self.config.n_layer) ** -0.5
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

    def _build_adamw(self, optim_groups, learning_rate, betas, eps, device_type):
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

    @staticmethod
    def _is_muon_parameter(name, param):
        if param.ndim != 2:
            return False
        name = name.removeprefix("_orig_mod.")
        hidden_weight_suffixes = (
            ".attn.c_attn.weight",
            ".attn.c_proj.weight",
            ".mlp.w1.weight",
            ".mlp.w2.weight",
            ".mlp.proj.weight",
        )
        return name.startswith("transformer.h.") and name.endswith(hidden_weight_suffixes)

    def _split_muon_params(self):
        muon_params = []
        adamw_decay_params = []
        adamw_nodecay_params = []
        seen = set()

        for name, param in self.named_parameters():
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            if self._is_muon_parameter(name, param):
                muon_params.append(param)
            elif param.ndim >= 2:
                adamw_decay_params.append(param)
            else:
                adamw_nodecay_params.append(param)

        return muon_params, adamw_decay_params, adamw_nodecay_params

    def configure_optimizers(
        self,
        weight_decay,
        learning_rate,
        betas,
        eps,
        device_type,
        optimizer="adamw",
        muon_lr=0.02,
        muon_momentum=0.95,
        muon_nesterov=True,
        muon_ns_steps=5,
    ):
        optimizer = optimizer.lower()
        if optimizer == "adamw":
            param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
            decay_params = [p for p in param_dict.values() if p.dim() >= 2]
            nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
            optim_groups = [
                {"params": decay_params, "weight_decay": weight_decay, "lr_scale": 1.0, "name": "adamw_decay"},
                {"params": nodecay_params, "weight_decay": 0.0, "lr_scale": 1.0, "name": "adamw_nodecay"},
            ]
            return self._build_adamw(optim_groups, learning_rate, betas, eps, device_type)

        if optimizer != "muon":
            raise ValueError(f"Unsupported optimizer {optimizer!r}; expected 'adamw' or 'muon'")

        muon_params, adamw_decay_params, adamw_nodecay_params = self._split_muon_params()
        adamw_groups = []
        if adamw_decay_params:
            adamw_groups.append(
                {"params": adamw_decay_params, "weight_decay": weight_decay, "lr_scale": 1.0, "name": "adamw_decay"}
            )
        if adamw_nodecay_params:
            adamw_groups.append(
                {"params": adamw_nodecay_params, "weight_decay": 0.0, "lr_scale": 1.0, "name": "adamw_nodecay"}
            )

        adamw = self._build_adamw(adamw_groups, learning_rate, betas, eps, device_type) if adamw_groups else None
        muon = Muon(
            [
                {
                    "params": muon_params,
                    "weight_decay": weight_decay,
                    "lr_scale": muon_lr / learning_rate,
                    "name": "muon",
                }
            ],
            lr=muon_lr,
            momentum=muon_momentum,
            nesterov=muon_nesterov,
            ns_steps=muon_ns_steps,
        )

        rank = int(os.environ.get("RANK", "0"))
        if rank == 0:
            muon_count = sum(p.numel() for p in muon_params)
            adamw_count = sum(p.numel() for p in adamw_decay_params) + sum(p.numel() for p in adamw_nodecay_params)
            print(
                f"optimizer: muon | muon tensors: {len(muon_params)} ({muon_count:,} params) | "
                f"adamw fallback tensors: {len(adamw_decay_params) + len(adamw_nodecay_params)} "
                f"({adamw_count:,} params)"
            )

        return CombinedOptimizer([muon, adamw])

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
    n_non_embedding = model.get_num_params(non_embedding=True)
    print(f"Parameters: {n_params:,}")
    print(f"Non-embedding parameters: {n_non_embedding:,}")
    assert n_params <= 100_000_000
    x = torch.randint(0, config.vocab_size, (1, min(16, config.block_size)))
    logits = model(x)
    print(f"Output shape: {tuple(logits.shape)}")
