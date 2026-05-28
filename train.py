import argparse
import importlib.util
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import GPT, GPTConfig


MODEL_CONFIG_KEYS = set(GPTConfig.__dataclass_fields__)


def repo_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def load_config(path):
    path = repo_path(path)
    spec = importlib.util.spec_from_file_location("train_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = {k: v for k, v in vars(module).items() if not k.startswith("_")}
    return SimpleNamespace(**values)


def config_to_dict(cfg):
    values = {}
    for key, value in vars(cfg).items():
        if key.startswith("_"):
            continue
        if isinstance(value, (str, int, float, bool, type(None), list, tuple, dict)):
            values[key] = value
    return values


def atomic_torch_save(obj, path):
    tmp_path = f"{path}.tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def clean_state_dict(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod.") :]
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    return cleaned


def is_full_training_checkpoint(checkpoint):
    return isinstance(checkpoint, dict) and "model" in checkpoint and "optimizer" in checkpoint


def get_model_state_from_checkpoint(checkpoint):
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def load_model_state(raw_model, checkpoint):
    state_dict = get_model_state_from_checkpoint(checkpoint)
    try:
        raw_model.load_state_dict(state_dict)
        return
    except RuntimeError:
        pass
    cleaned = clean_state_dict(state_dict)
    if hasattr(raw_model, "_orig_mod"):
        raw_model._orig_mod.load_state_dict(cleaned)
    else:
        raw_model.load_state_dict(cleaned)


def load_tokens(filename):
    if filename.endswith(".npy"):
        return np.load(filename, mmap_mode="r")
    if filename.endswith(".bin"):
        return np.memmap(filename, dtype=np.uint16, mode="r")
    raise ValueError(f"Unsupported token shard format: {filename}")


class ShardedTokenLoader:
    def __init__(self, B, T, process_rank, num_processes, split, data_dir, allow_val_train_fallback=False):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split
        self.data_dir = data_dir
        shards = self._find_shards(split)
        if not shards and split == "val" and allow_val_train_fallback:
            shards = self._find_shards("train")
            print(f"No val shards found in {data_dir}; using train shards for validation.")
        if not shards:
            raise FileNotFoundError(f"No {split} shards found in {data_dir}")
        self.shards = shards
        self.reset()

    def _find_shards(self, split):
        suffixes = (".bin", ".npy")
        return sorted(
            os.path.join(self.data_dir, name)
            for name in os.listdir(self.data_dir)
            if split in name and name.endswith(suffixes)
        )

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank
        self._ensure_enough_tokens()

    def _advance_shard(self):
        self.current_shard = (self.current_shard + 1) % len(self.shards)
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def _ensure_enough_tokens(self):
        attempts = 0
        while self.current_position + self.B * self.T + 1 > len(self.tokens):
            self._advance_shard()
            attempts += 1
            if attempts > len(self.shards):
                needed = self.current_position + self.B * self.T + 1
                raise ValueError(f"No {self.split} shard in {self.data_dir} has enough tokens for {needed=}")

    def next_batch(self):
        B, T = self.B, self.T
        self._ensure_enough_tokens()
        buf = torch.from_numpy(
            self.tokens[self.current_position : self.current_position + B * T + 1].astype(np.int64)
        )
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self._advance_shard()
        return x, y

    def state_dict(self):
        return {
            "type": "sharded",
            "split": self.split,
            "data_dir": self.data_dir,
            "shards": list(self.shards),
            "current_shard": self.current_shard,
            "current_position": self.current_position,
        }

    def load_state_dict(self, state):
        if state.get("type") != "sharded":
            raise ValueError(f"Expected sharded loader state, got {state.get('type')!r}")
        if state.get("split") != self.split:
            raise ValueError(f"Loader split mismatch: checkpoint={state.get('split')!r}, current={self.split!r}")
        if os.path.abspath(state.get("data_dir", "")) != os.path.abspath(self.data_dir):
            raise ValueError("Loader data_dir mismatch; use --reset-loader-state to ignore saved loader position")
        if list(state.get("shards", [])) != list(self.shards):
            raise ValueError("Training shard list changed; use --reset-loader-state to start loaders from the beginning")
        self.current_shard = int(state["current_shard"])
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = int(state["current_position"])
        self._ensure_enough_tokens()


class TokenFileLoader:
    def __init__(self, B, T, process_rank, num_processes, filename, label="tokens"):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.filename = filename
        self.label = label
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Missing {label} token file: {filename}")
        self.reset()

    def reset(self):
        self.tokens = load_tokens(self.filename)
        self.current_position = self.B * self.T * self.process_rank
        self._ensure_enough_tokens()

    def _ensure_enough_tokens(self):
        needed = self.current_position + self.B * self.T + 1
        if needed > len(self.tokens):
            if self.current_position == self.B * self.T * self.process_rank:
                raise ValueError(f"{self.label} file {self.filename} is too small for {needed=}")
            self.current_position = self.B * self.T * self.process_rank
            needed = self.current_position + self.B * self.T + 1
            if needed > len(self.tokens):
                raise ValueError(f"{self.label} file {self.filename} is too small for {needed=}")

    def next_batch(self):
        B, T = self.B, self.T
        self._ensure_enough_tokens()
        buf = torch.from_numpy(
            self.tokens[self.current_position : self.current_position + B * T + 1].astype(np.int64)
        )
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank
        return x, y

    def state_dict(self):
        return {
            "type": "token_file",
            "filename": self.filename,
            "current_position": self.current_position,
        }

    def load_state_dict(self, state):
        if state.get("type") != "token_file":
            raise ValueError(f"Expected token_file loader state, got {state.get('type')!r}")
        if os.path.abspath(state.get("filename", "")) != os.path.abspath(self.filename):
            raise ValueError("Validation file mismatch; use --reset-loader-state to ignore saved loader position")
        self.current_position = int(state["current_position"])
        self._ensure_enough_tokens()


class WeightedSource:
    def __init__(self, name, loaders, weights=None):
        self.name = name
        self.loaders = loaders
        self.weights = list(weights or [1.0] * len(loaders))
        if len(self.loaders) != len(self.weights):
            raise ValueError(f"{name} has {len(loaders)} loaders but {len(weights)} weights")
        if not self.loaders:
            raise ValueError(f"{name} must have at least one loader")
        if any(weight <= 0 for weight in self.weights):
            raise ValueError(f"{name} weights must be positive")
        self.current_weights = [0.0] * len(self.loaders)

    def next_batch(self):
        if len(self.loaders) == 1:
            return self.loaders[0].next_batch()

        total_weight = sum(self.weights)
        for i, weight in enumerate(self.weights):
            self.current_weights[i] += weight
        loader_index = max(range(len(self.loaders)), key=lambda i: self.current_weights[i])
        self.current_weights[loader_index] -= total_weight
        return self.loaders[loader_index].next_batch()

    def state_dict(self):
        return {
            "name": self.name,
            "weights": list(self.weights),
            "current_weights": list(self.current_weights),
            "loaders": [loader.state_dict() for loader in self.loaders],
        }

    def load_state_dict(self, state):
        if state.get("name") != self.name:
            raise ValueError(f"Source name mismatch: checkpoint={state.get('name')!r}, current={self.name!r}")
        if list(state.get("weights", [])) != list(self.weights):
            raise ValueError("Source weights changed; use --reset-loader-state to ignore saved loader position")
        loader_states = state.get("loaders", [])
        if len(loader_states) != len(self.loaders):
            raise ValueError("Source loader count changed; use --reset-loader-state to ignore saved loader position")
        self.current_weights = list(state["current_weights"])
        for loader, loader_state in zip(self.loaders, loader_states):
            loader.load_state_dict(loader_state)


class MixedTokenLoader:
    def __init__(self, B, T, process_rank, num_processes, mix_specs):
        self.schedule = []
        self.sources = []
        for spec in mix_specs:
            source = self._build_source(B, T, process_rank, num_processes, spec)
            self.sources.append(source)
            self.schedule.extend([source] * int(spec["micro_batches"]))
        if not self.schedule:
            raise ValueError("train_data_mix must contain at least one scheduled microbatch")

    def _build_source(self, B, T, process_rank, num_processes, spec):
        name = spec["name"]
        if "subsets" in spec:
            loaders = []
            weights = []
            for subset in spec["subsets"]:
                data_dir = repo_path(subset["data_dir"])
                loaders.append(ShardedTokenLoader(B, T, process_rank, num_processes, "train", data_dir))
                weights.append(float(subset.get("weight", 1.0)))
            return WeightedSource(name, loaders, weights)

        data_dir = repo_path(spec["data_dir"])
        loader = ShardedTokenLoader(B, T, process_rank, num_processes, "train", data_dir)
        return WeightedSource(name, [loader])

    def next_batch(self, micro_step):
        source = self.schedule[micro_step % len(self.schedule)]
        return source.next_batch()

    def describe(self):
        counts = {}
        for source in self.schedule:
            counts[source.name] = counts.get(source.name, 0) + 1
        return ", ".join(f"{name}={count}" for name, count in counts.items())

    def state_dict(self):
        return {
            "type": "mixed",
            "schedule_names": [source.name for source in self.schedule],
            "sources": [source.state_dict() for source in self.sources],
        }

    def load_state_dict(self, state):
        if state.get("type") != "mixed":
            raise ValueError(f"Expected mixed loader state, got {state.get('type')!r}")
        schedule_names = [source.name for source in self.schedule]
        if list(state.get("schedule_names", [])) != schedule_names:
            raise ValueError("Training mix schedule changed; use --reset-loader-state to ignore saved loader position")
        source_states = state.get("sources", [])
        if len(source_states) != len(self.sources):
            raise ValueError("Training source count changed; use --reset-loader-state to ignore saved loader position")
        for source, source_state in zip(self.sources, source_states):
            source.load_state_dict(source_state)


def setup_ddp():
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "DDP training requires CUDA"
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        return True, rank, local_rank, world_size, device, rank == 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    return False, 0, 0, 1, device, True


def get_lr(step, cfg):
    if step < cfg.warmup_steps:
        return cfg.max_lr * (step + 1) / cfg.warmup_steps
    if step > cfg.max_steps:
        return cfg.min_lr
    decay_ratio = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.max_lr - cfg.min_lr)


def autocast_context(device_type, cfg):
    if not cfg.use_amp or device_type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16
    return torch.autocast(device_type=device_type, dtype=dtype)


@torch.no_grad()
def estimate_val_loss(model, loader, cfg, device, device_type, ddp):
    model.eval()
    loader.reset()
    loss_accum = torch.zeros((), device=device)
    for _ in range(cfg.eval_iters):
        x, y = loader.next_batch()
        x, y = x.to(device), y.to(device)
        with autocast_context(device_type, cfg):
            _, loss = model(x, y)
        loss_accum += loss / cfg.eval_iters
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    model.train()
    return loss_accum.item()


def restore_rng_state(state):
    if not state:
        return
    torch_state = state.get("torch")
    if torch_state is not None:
        torch.set_rng_state(torch_state.cpu())
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_state])
    numpy_state = state.get("numpy")
    if numpy_state is not None:
        np.random.set_state(numpy_state)


def get_rng_state():
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
    }


def validate_resume_config(checkpoint, cfg, reset_optimizer=False, reset_loader_state=False):
    saved_model_config = checkpoint.get("model_config", {})
    current_config = config_to_dict(cfg)
    for key in sorted(MODEL_CONFIG_KEYS):
        if key in saved_model_config and key in current_config and saved_model_config[key] != current_config[key]:
            raise ValueError(
                f"Cannot resume: model config {key} differs "
                f"(checkpoint={saved_model_config[key]!r}, current={current_config[key]!r})"
            )

    saved_train_config = checkpoint.get("train_config", {})
    if not reset_optimizer and saved_train_config.get("optimizer", "adamw") != current_config.get("optimizer", "adamw"):
        raise ValueError("Optimizer changed; use --reset-optimizer to resume with a fresh optimizer")

    if not reset_loader_state:
        for key in ("data_dir", "val_data_path", "train_data_mix"):
            if key in saved_train_config and key in current_config and saved_train_config[key] != current_config[key]:
                raise ValueError(f"Data config {key} changed; use --reset-loader-state to ignore saved loader state")


def save_checkpoint_metadata(checkpoint_dir, filename, model_config, step=None, val_loss=None, best_val_loss=None):
    os.makedirs(checkpoint_dir, exist_ok=True)
    metadata = {
        "model_config": asdict(model_config),
        "step": step,
        "global_step": step,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
    }
    metadata_name = filename.replace(".pt", "_metadata.pt")
    atomic_torch_save(metadata, os.path.join(checkpoint_dir, metadata_name))


def save_training_checkpoint(
    raw_model,
    optimizer,
    checkpoint_dir,
    filename,
    model_config,
    cfg,
    global_step,
    val_loss,
    best_val_loss,
    train_loader,
    val_loader,
    grad_accum_steps,
    world_size,
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    payload = {
        "version": 1,
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": asdict(model_config),
        "train_config": config_to_dict(cfg),
        "global_step": global_step,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "loader_state": {
            "train": train_loader.state_dict(),
            "val": val_loader.state_dict(),
        },
        "rng_state": get_rng_state(),
        "runtime": {
            "grad_accum_steps": grad_accum_steps,
            "world_size": world_size,
            "saved_time": time.time(),
        },
    }
    path = os.path.join(checkpoint_dir, filename)
    atomic_torch_save(payload, path)
    save_checkpoint_metadata(checkpoint_dir, filename, model_config, global_step, val_loss, best_val_loss)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.py")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--reset-loader-state", action="store_true")
    parser.add_argument("--reset-optimizer", action="store_true")
    parser.add_argument("--reset-rng", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    resume_path = args.resume or getattr(cfg, "resume_from", None)
    steps_this_run = args.steps or getattr(cfg, "steps_this_run", None)

    ddp, rank, local_rank, world_size, device, master_process = setup_ddp()
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    torch.manual_seed(cfg.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed + rank)
    torch.set_float32_matmul_precision(cfg.matmul_precision)

    data_dir = repo_path(cfg.data_dir)
    checkpoint_dir = repo_path(cfg.checkpoint_dir)
    log_file = repo_path(cfg.log_file)

    B = cfg.micro_batch_size
    T = cfg.block_size
    assert cfg.total_batch_size % (B * T * world_size) == 0
    grad_accum_steps = cfg.total_batch_size // (B * T * world_size)
    if master_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        if resume_path:
            with open(log_file, "a") as f:
                f.write(f"resume {resume_path}\n")
        else:
            with open(log_file, "w") as f:
                f.write("")
        print(f"device: {device}")
        print(f"gradient accumulation steps: {grad_accum_steps}")

    train_mix = getattr(cfg, "train_data_mix", None)
    if train_mix:
        train_loader = MixedTokenLoader(B, T, rank, world_size, train_mix)
        if len(train_loader.schedule) != grad_accum_steps:
            raise ValueError(
                f"train_data_mix schedules {len(train_loader.schedule)} microbatches, "
                f"but grad_accum_steps is {grad_accum_steps}"
            )
        if master_process:
            print(f"train data mix per optimizer step: {train_loader.describe()}")
    else:
        train_loader = ShardedTokenLoader(B, T, rank, world_size, "train", data_dir)

    val_data_path = getattr(cfg, "val_data_path", None)
    if val_data_path:
        val_loader = TokenFileLoader(B, T, rank, world_size, repo_path(val_data_path), label="validation")
        if master_process:
            print(f"validation data: {repo_path(val_data_path)}")
    else:
        allow_val_train_fallback = getattr(cfg, "allow_val_train_fallback", False)
        val_loader = ShardedTokenLoader(B, T, rank, world_size, "val", data_dir, allow_val_train_fallback)

    model_config = GPTConfig(
        block_size=cfg.block_size,
        vocab_size=cfg.vocab_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        mlp_hidden_dim=getattr(cfg, "mlp_hidden_dim", 4 * cfg.n_embd),
        dropout=getattr(cfg, "dropout", 0.0),
        bias=getattr(cfg, "bias", False),
    )
    model = GPT(model_config).to(device)
    if master_process:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"parameters: {n_params:,}")
        if n_params > 100_000_000:
            raise ValueError(f"Model has {n_params:,} parameters, above the 100M limit")

    if cfg.compile_model:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[local_rank])
    raw_model = model.module if ddp else model

    optimizer = raw_model.configure_optimizers(
        weight_decay=cfg.weight_decay,
        learning_rate=cfg.max_lr,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        device_type=device_type,
        optimizer=getattr(cfg, "optimizer", "adamw"),
        muon_lr=getattr(cfg, "muon_lr", 0.02),
        muon_momentum=getattr(cfg, "muon_momentum", 0.95),
        muon_nesterov=getattr(cfg, "muon_nesterov", True),
        muon_ns_steps=getattr(cfg, "muon_ns_steps", 5),
    )

    full_resume = False
    start_step = 0
    best_val_loss = float("inf")
    if resume_path:
        checkpoint = torch_load(repo_path(resume_path), device)
        load_model_state(raw_model, checkpoint)
        full_resume = is_full_training_checkpoint(checkpoint)
        if full_resume:
            validate_resume_config(
                checkpoint,
                cfg,
                reset_optimizer=args.reset_optimizer,
                reset_loader_state=args.reset_loader_state,
            )
            if not args.reset_optimizer:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if not args.reset_loader_state:
                loader_state = checkpoint.get("loader_state", {})
                train_loader.load_state_dict(loader_state["train"])
                val_loader.load_state_dict(loader_state["val"])
            if not args.reset_rng:
                restore_rng_state(checkpoint.get("rng_state"))
            start_step = int(checkpoint.get("global_step", -1)) + 1
            best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        if master_process:
            resume_kind = "full checkpoint" if full_resume else "warm-start weights"
            print(f"resumed {resume_kind} from {repo_path(resume_path)}")
            print(f"start step: {start_step}")
            if full_resume:
                print(f"best val loss so far: {best_val_loss:.6f}")
            with open(log_file, "a") as f:
                f.write(f"resume_kind {resume_kind} start_step {start_step}\n")

    end_step = cfg.max_steps if steps_this_run is None else start_step + int(steps_this_run)
    if end_step <= start_step:
        raise ValueError(f"No training steps requested: start_step={start_step}, end_step={end_step}")
    checkpoint_filename = getattr(cfg, "checkpoint_filename", "checkpoint.pt")
    if master_process:
        print(f"training global steps: {start_step}..{end_step - 1}")

    for step in range(start_step, end_step):
        t0 = time.time()
        last_step = step == end_step - 1

        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_accum = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if train_mix:
                x, y = train_loader.next_batch(micro_step)
            else:
                x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            if ddp:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            with autocast_context(device_type, cfg):
                _, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        lr = get_lr(step, cfg)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * param_group.get("lr_scale", 1.0)
        optimizer.step()
        if device_type == "cuda":
            torch.cuda.synchronize()

        if master_process and (step % cfg.log_interval == 0 or last_step):
            dt = time.time() - t0
            tokens_processed = B * T * grad_accum_steps * world_size
            tok_per_sec = tokens_processed / dt
            print(
                f"step {step:5d} | train loss {loss_accum.item():.6f} | "
                f"lr {lr:.4e} | norm {norm:.4f} | {tok_per_sec:.0f} tok/s"
            )
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")

        if step % cfg.eval_interval == 0 or last_step:
            val_loss = estimate_val_loss(model, val_loader, cfg, device, device_type, ddp)
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            if master_process:
                suffix = " | saved checkpoint.pt" if is_best else ""
                print(f"step {step:5d} | val loss {val_loss:.4f} | best {best_val_loss:.4f}{suffix}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss:.6f} best {best_val_loss:.6f}\n")
                if is_best:
                    save_training_checkpoint(
                        raw_model,
                        optimizer,
                        checkpoint_dir,
                        checkpoint_filename,
                        model_config,
                        cfg,
                        step,
                        val_loss,
                        best_val_loss,
                        train_loader,
                        val_loader,
                        grad_accum_steps,
                        world_size,
                    )
                    if getattr(cfg, "save_step_checkpoints", False):
                        save_training_checkpoint(
                            raw_model,
                            optimizer,
                            checkpoint_dir,
                            f"checkpoint_{step:05d}.pt",
                            model_config,
                            cfg,
                            step,
                            val_loss,
                            best_val_loss,
                            train_loader,
                            val_loader,
                            grad_accum_steps,
                            world_size,
                        )

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
