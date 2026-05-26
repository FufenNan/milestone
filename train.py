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


def save_checkpoint(raw_model, checkpoint_dir, filename, model_config, step=None, val_loss=None):
    os.makedirs(checkpoint_dir, exist_ok=True)
    state_dict = raw_model.state_dict()
    torch.save(state_dict, os.path.join(checkpoint_dir, filename))
    metadata = {"model_config": asdict(model_config), "step": step, "val_loss": val_loss}
    torch.save(metadata, os.path.join(checkpoint_dir, filename.replace(".pt", "_metadata.pt")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.py")
    args = parser.parse_args()
    cfg = load_config(args.config)

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

    for step in range(cfg.max_steps):
        t0 = time.time()
        last_step = step == cfg.max_steps - 1

        if step % cfg.eval_interval == 0 or last_step:
            val_loss = estimate_val_loss(model, val_loader, cfg, device, device_type, ddp)
            if master_process:
                print(f"step {step:5d} | val loss {val_loss:.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss:.6f}\n")
                should_save = step > 0 and (step % cfg.checkpoint_interval == 0 or last_step)
                if should_save:
                    checkpoint_filename = getattr(cfg, "checkpoint_filename", "checkpoint.pt")
                    save_checkpoint(raw_model, checkpoint_dir, checkpoint_filename, model_config, step, val_loss)
                    if cfg.save_step_checkpoints:
                        save_checkpoint(raw_model, checkpoint_dir, f"checkpoint_{step:05d}.pt", model_config, step, val_loss)

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

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
