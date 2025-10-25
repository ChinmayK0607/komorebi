#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, os, time, math, glob, random
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# Optional wandb (graceful fallback)
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False
    wandb = None  # type: ignore


# -------------------------
# Binary token loaders (uint16 GPT-2 ids)
# -------------------------
def list_split_bins(data_dir: str, split: str, num_chunks: Optional[int]) -> List[str]:
    """
    Finds split files:
      train: fineweb_train_%06d.bin
      val:   fineweb_val_%06d.bin
    If num_chunks is given, truncates the list (useful for quick tests).
    """
    pattern = os.path.join(data_dir, f"fineweb_{split}_*.bin")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No {split} .bin files found under {data_dir} (looked for {pattern}).")
    if num_chunks is not None:
        files = files[:num_chunks]
    return files


class BinChunkSampler:
    """
    Wraps multiple uint16 memmaps and provides random contiguous slices of length (T+1)
    suitable for next-token prediction batches.

    Sampling is weighted by each file's available start positions (len - (T+1)).
    """
    def __init__(self, files: List[str], T: int):
        self.T = T
        self.arrs: List[np.memmap] = []
        self.lengths: List[int] = []
        self.starts_cap: List[int] = []  # how many valid start positions per file
        for f in files:
            m = np.memmap(f, dtype=np.uint16, mode="r")
            self.arrs.append(m)
            L = int(m.shape[0])
            self.lengths.append(L)
            cap = max(0, L - (T + 1))
            self.starts_cap.append(cap)
        self.total_cap = sum(self.starts_cap)
        if self.total_cap <= 0:
            raise ValueError("No valid start positions across files (increase T or use larger files).")
        # cumulative weights for weighted file sampling
        self.cum_caps = np.cumsum(self.starts_cap)

    def _sample_file_index(self) -> int:
        r = random.randrange(self.total_cap)  # [0, total_cap)
        # binary search cumulative caps
        return int(np.searchsorted(self.cum_caps, r, side="right"))
    def _sample_file_index_with_rng(self, rng: random.Random) -> int:
        r = rng.randrange(self.total_cap)
        return int(np.searchsorted(self.cum_caps, r, side="right"))

    def sample_one(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return one (x, y) pair of shape (T,) from a random file and start."""
        fi = self._sample_file_index()
        L = self.lengths[fi]
        start = random.randint(0, L - (self.T + 1))
        buf = self.arrs[fi][start : start + self.T + 1].astype(np.int64)  # upcast to fit vocab indices
        x = buf[:-1]
        y = buf[1:]
        return x, y


class RandomBatchLoader:
    def __init__(self, data_dir, split, B, T, num_chunks, seed=1337, rank=0):
        # in RandomBatchLoader.__init__
        

        files = list_split_bins(data_dir, split, num_chunks)
        self.sampler = BinChunkSampler(files, T)
        self.B, self.T = B, T
        self.rng = random.Random(seed + rank)   # rank-sharded RNG
        if rank == 0:
            print(f"[{split}] rank={rank} files={len(files)} tokens={sum(self.sampler.lengths):,} T={T}")

    def next_batch(self, device):
        xs, ys = [], []
        for _ in range(self.B):
            # use self.rng instead of global random
            fi = self.sampler._sample_file_index_with_rng(self.rng)
            L  = self.sampler.lengths[fi]
            start = self.rng.randint(0, L - (self.T + 1))
            buf = self.sampler.arrs[fi][start:start + self.T + 1].astype(np.int64)
            xs.append(buf[:-1]); ys.append(buf[1:])
        x = torch.tensor(np.stack(xs), dtype=torch.long, device=device)
        y = torch.tensor(np.stack(ys), dtype=torch.long, device=device)
        return x, y



# -------------------------
# Model
# -------------------------
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257  # GPT-2 vocab
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embed % cfg.n_head == 0
        self.n_embed, self.n_head = cfg.n_embed, cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embed, 3 * cfg.n_embed)
        self.c_proj = nn.Linear(cfg.n_embed, cfg.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        y = self.resid_drop(y)
        return y


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embed, 4 * cfg.n_embed)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * cfg.n_embed, cfg.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x); x = self.gelu(x); x = self.c_proj(x); x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embed)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embed)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tr = nn.ModuleDict(
            dict(
                wte=nn.Embedding(cfg.vocab_size, cfg.n_embed),
                wpe=nn.Embedding(cfg.block_size, cfg.n_embed),
                h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
                ln_f=nn.LayerNorm(cfg.n_embed),
            )
        )
        self.lm_head = nn.Linear(cfg.n_embed, cfg.vocab_size, bias=False)
        # weight tying
        self.tr.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            std = 0.02
            if hasattr(m, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.cfg.n_layer) ** -0.5
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.size()
        if T > self.cfg.block_size:
            raise ValueError(f"Sequence length {T} > block_size {self.cfg.block_size}")
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.tr.wte(idx) + self.tr.wpe(pos)[None, :, :]
        for blk in self.tr.h:
            x = blk(x)
        x = self.tr.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# -------------------------
# Eval helper
# -------------------------
@torch.no_grad()
def estimate_loss(model, loader, eval_iters, device, amp_mode: str) -> float:
    model.eval()
    losses = []
    use_amp = amp_mode in ("fp16", "bf16") and device.type == "cuda"
    dtype = torch.float16 if amp_mode == "fp16" else torch.bfloat16
    ctx = (torch.autocast(device_type=device.type, dtype=dtype)
           if use_amp else torch.cuda.amp.autocast(enabled=False))
    with ctx:
        for _ in range(eval_iters):
            x, y = loader.next_batch(device)
            _, loss = model(x, y)
            losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)



# -------------------------
# Main
# -------------------------
def main():
    p = argparse.ArgumentParser(description="GPT trainer on FineWeb10B .bin tokens (AMP + grad accumulation + wandb)")

    # data / batching
    p.add_argument("--data_dir", type=str, default="fineweb10B", help="Directory containing fineweb_*_*.bin")
    p.add_argument("--num_train_chunks", type=int, default=None, help="Limit number of train bins (debug)")
    p.add_argument("--num_val_chunks", type=int, default=1, help="Limit number of val bins (e.g., 1 for fineweb_val_000000.bin)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--block_size", type=int, default=512)
    p.add_argument("--eval_interval", type=int, default=1000)
    p.add_argument("--eval_iters", type=int, default=200)

    # model
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_embed", type=int, default=786)
    p.add_argument("--dropout", type=float, default=0.2)

    # optimization
    p.add_argument("--max_iters", type=int, default=50000)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--grad_accum_steps", type=int, default=1)

    # AMP
    p.add_argument("--amp", type=str, choices=["none", "fp16", "bf16"], default="none")

    # device / seed
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--seed", type=int, default=1337)

    # wandb
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="fineweb10B-gpt")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--ddp", action="store_true", help="Enable DistributedDataParallel")
    p.add_argument("--dist_backend", type=str, default="nccl", choices=["nccl","gloo"])
    p.add_argument("--save_interval", type=int, default=0, help="0=disable periodic checkpointing")

    args = p.parse_args()
    import torch.distributed as dist

    def is_dist():
        return dist.is_available() and dist.is_initialized()

    def get_rank():
        return dist.get_rank() if is_dist() else 0

    def get_world_size():
        return dist.get_world_size() if is_dist() else 1

    # --- initialize DDP if requested ---
    if args.ddp:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA in this script.")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend=args.dist_backend)
    else:
        if args.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(args.device)


    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    global_seed = args.seed
    rank = get_rank()
    global_seed = args.seed + rank
    random.seed(global_seed)
    np.random.seed(global_seed)
    torch.manual_seed(global_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(global_seed)


    # loaders
    train_loader = RandomBatchLoader(args.data_dir, "train", args.batch_size, args.block_size,
                                 args.num_train_chunks, seed=args.seed, rank=rank)
    val_loader   = RandomBatchLoader(args.data_dir, "val",   args.batch_size, args.block_size,
                                 args.num_val_chunks,   seed=args.seed+999, rank=rank)


    # model + optimizer
    cfg = GPTConfig(
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embed=args.n_embed,
        dropout=args.dropout,
    )
    from torch.nn.parallel import DistributedDataParallel as DDP

    model = GPT(cfg).to(device)

    if args.ddp:
        # important: pass device_ids + broadcast_buffers off for transformers
        model = DDP(model, device_ids=[device.index], output_device=device.index, broadcast_buffers=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=tuple(args.betas),
        weight_decay=args.weight_decay,
    )

    # AMP
    use_amp = args.amp in ("fp16", "bf16") and device.type in ("cuda", "mps")
    amp_dtype = torch.float16 if args.amp == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler('cuda',enabled=(args.amp == "fp16" and device.type == "cuda"))
    is_main = (rank == 0)

    def log_wandb(payload, step=None):
        if is_main and run is not None:
            wandb.log(payload, step=step)

    if is_main:
        print(f"DDP={args.ddp} world_size={get_world_size()} device={device}")

    # wandb
    run = None
    if args.wandb and WANDB_AVAILABLE and args.wandb_mode != "disabled":
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity if args.wandb_entity else None,
            config={**vars(args), **asdict(cfg), "device": device.type},
            mode=args.wandb_mode,
            name=args.run_name if args.run_name else None,
        )
    elif args.wandb and not WANDB_AVAILABLE:
        print("wandb not available; proceeding without external logging.")

    # training
    model.train()
    t0 = time.time()
    tokens_per_micro = args.batch_size * args.block_size
    micro_in_macro = args.grad_accum_steps
    optimizer.zero_grad(set_to_none=True)
    last_log_time = time.time()

    for step in range(1, args.max_iters + 1):
        # accumulate
        for micro in range(1, micro_in_macro + 1):
            x, y = train_loader.next_batch(device)
            with (torch.autocast(device_type=device.type, dtype=amp_dtype) if use_amp
                else torch.amp.autocast('cuda',enabled=False)):
                _, loss = model(x, y)
                loss = loss / micro_in_macro

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

        # average last-micro loss across ranks for logging
        loss_scalar = loss.detach().float()
        if args.ddp:
            dist.all_reduce(loss_scalar, op=dist.ReduceOp.AVG)

        # clip (ALL RANKS)
        if args.grad_clip > 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        # step (ALL RANKS)
        if scaler.is_enabled():
            scaler.step(optimizer); scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # throughput (rank-0 prints)
        now = time.time()
        dt = now - last_log_time
        eff_toks_per_sec = (tokens_per_micro * micro_in_macro) / dt if dt > 0 else float("inf")
        last_log_time = now

        if is_main and ((step % max(1, args.eval_interval // 10)) == 0 or step == 1):
            print(f"step {step:6d} | train_loss (global last micro) {loss_scalar.item():.4f} | eff_tok/s {eff_toks_per_sec:,.0f}")

        if is_main and run:
            wandb.log({
                "train/loss_last_micro_global": float(loss_scalar.item()),
                "opt/lr": float(optimizer.param_groups[0]["lr"]),
                "speed/eff_tokens_per_sec": eff_toks_per_sec,
                "amp/enabled": int(use_amp),
                "amp/mode": args.amp,
                "accum/steps": args.grad_accum_steps,
                "step": step,
            }, step=step)

        # periodic eval
        if (step % args.eval_interval) == 0 or step == args.max_iters:
            eval_loss = estimate_loss(model, val_loader, args.eval_iters, device, args.amp)
            tensor = torch.tensor([eval_loss], device=device)
            if args.ddp:
                dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
            eval_loss_global = float(tensor.item())
            if is_main:
                print(f"[eval] step {step} | eval_loss {eval_loss_global:.4f} | elapsed {time.time()-t0:.1f}s")
                if run:
                    wandb.log({"eval/loss": eval_loss_global, "step": step}, step=step)



        if args.save_interval and (step % args.save_interval == 0) and is_main:
            to_save = model.module if args.ddp else model
            torch.save({"model": to_save.state_dict(), "cfg": asdict(cfg), "step": step},
                    f"ckpt_{step:07d}.pt")
    if is_main and run:
        run.finish()
        print("training finished.")

    if args.ddp:
        dist.barrier()
        dist.destroy_process_group()



if __name__ == "__main__":
    main()
