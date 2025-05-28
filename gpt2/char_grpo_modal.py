#!/usr/bin/env python
"""
GRPO on tiny-Shakespeare with a frozen bigram reference.
Vectorised roll-outs + partial-log-prob shaping.

Launch: modal run grpo_bigram_ref.py
"""

# ── hyper-params ─────────────────────────────────────────────────────────
HORIZON        = 10    # length of each rollout sequence
BATCH          = 64
K_SAMPLES      = 70     # rollouts per prompt (vectorised)
TOTAL_ITERS    = 2000
LR             = 3e-5
BETA_KL        = 3e-3
KL_WARM        = 20_000
NEG_REWARD     = False
PARTIAL_LP     = 0.05
GPU_TYPE       = "A10G"   # "A100", "L40S", …

# ── Modal plumbing ───────────────────────────────────────────────────────
import math, requests, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import modal, wandb, numpy as np, copy

stub  = modal.App("grpo-char-bigram")
image = (
    modal.Image.debian_slim()
    .pip_install("numpy", "torch==2.3.0", "tqdm", "wandb", "requests")
)

DATA_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
            "tinyshakespeare/input.txt")

def ensure_dataset(fname="input.txt"):
    if not Path(fname).exists():
        Path(fname).write_text(
            requests.get(DATA_URL, timeout=10).text, encoding="utf-8"
        )

def build_vocab(fname="input.txt"):
    text  = Path(fname).read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi  = {ch: i for i, ch in enumerate(chars)}
    itos  = {i: ch for ch, i in stoi.items()}
    def enc(s): return torch.tensor([stoi[c] for c in s], dtype=torch.long)
    def dec(t): return "".join(itos[int(i)] for i in t)
    return enc, dec, len(chars), stoi, text

class TextLoader:
    def __init__(self, toks, B, T):
        self.data, self.B, self.T, self.pos = toks, B, T, 0
    def next(self):
        span = self.B * self.T + 1
        if self.pos + span > len(self.data): self.pos = 0
        buf = self.data[self.pos : self.pos + span]; self.pos += span
        return buf[:-1].view(self.B, self.T), buf[1:].view(self.B, self.T)

# ── tiny GPT actor ───────────────────────────────────────────────────────
class TinyGPT(nn.Module):
    def __init__(self, V, n_layer=6, n_head=6, n_emb=384, blk=256):
        super().__init__()
        self.wte = nn.Embedding(V, n_emb); self.wpe = nn.Embedding(blk, n_emb)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                n_emb, n_head, 4*n_emb, activation="gelu", batch_first=True
            ) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, V, bias=False); self.head.weight = self.wte.weight
    def forward(self, idx):
        B, T = idx.shape; pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for lyr in self.layers: x = lyr(x)
        return self.head(self.ln_f(x))               # (B,T,V)

# ── frozen bigram reference ──────────────────────────────────────────────
class BigramRef(nn.Module):
    def __init__(self, counts):
        super().__init__()
        probs = (counts + 1) / (counts.sum(1, keepdims=True) + counts.size(1))
        self.register_buffer("logits", torch.log(probs))
    def forward(self, idx):                         # idx: (B,T)
        return self.logits[idx]                     # (B,T,V)

# ── helpers ──────────────────────────────────────────────────────────────
def tok_reward(gen, ref, neg=False):
    m = (gen == ref).float()
    return m*2 - 1 if neg else m

def add_partial(r, lp, coeff):
    return r if coeff == 0 else r + coeff * lp.detach()

@torch.no_grad()
def generate(model, ctx, steps):
    for _ in range(steps):
        nxt = torch.multinomial(torch.softmax(model(ctx)[:, -1], -1), 1)
        ctx = torch.cat([ctx, nxt], 1)
    return ctx[:, -steps:]         # (B,T)

# ── training loop ────────────────────────────────────────────────────────
@stub.function(
    gpu=GPU_TYPE, image=image, timeout=3*60*60,
    secrets=[modal.Secret.from_name("wandb")]
)
def train_remote():
    torch.set_float32_matmul_precision("high")
    DEV = "cuda" if torch.cuda.is_available() else "cpu"

    ensure_dataset()
    ENC, DEC, V, stoi, text = build_vocab()
    loader = TextLoader(ENC(text), BATCH, HORIZON + 1)

    # bigram reference
    counts = torch.zeros(V, V)
    for a, b in zip(text, text[1:]): counts[stoi[a], stoi[b]] += 1
    ref = BigramRef(counts).to(DEV).eval()

    actor = TinyGPT(V).to(DEV)
    opt   = torch.optim.AdamW(actor.parameters(), lr=LR)

    run = wandb.init(
        project="gpt2-grpo-char",
        config=dict(horizon=HORIZON, beta=BETA_KL, k=K_SAMPLES, partial_lp=PARTIAL_LP)
    )

    chars_seen = 0
    for it in tqdm(range(1, TOTAL_ITERS + 1), leave=False, desc="iters"):
        # ---- prompt & ground truth ------------------------------------
        ctx0, ref_tok = loader.next()
        ctx0, ref_tok = ctx0[:, :1].to(DEV), ref_tok[:, :HORIZON].to(DEV)

        # ---- vectorised K roll-outs -----------------------------------
        prompt = ctx0.repeat_interleave(K_SAMPLES, 0)          # (B*K,1)
        gen    = generate(actor, prompt, HORIZON)              # (B*K,T)
        logits = actor(gen)                                    # (B*K,T,V)
        dist   = torch.distributions.Categorical(logits.softmax(-1))
        logp   = dist.log_prob(gen)                            # (B*K,T)

        # reshape back to (B,K,T)
        gen   = gen.view(BATCH, K_SAMPLES, HORIZON)
        logp  = logp.view(BATCH, K_SAMPLES, HORIZON)

        r_tok = tok_reward(gen, ref_tok.unsqueeze(1), NEG_REWARD)
        r_tok = add_partial(r_tok, logp, PARTIAL_LP)           # (B,K,T)

        # ---- accuracy metrics ----------------------------------------
        correct = (gen == ref_tok.unsqueeze(1))
        rollout_correct = correct.sum(-1).float()              # (B,K)
        rollout_acc     = rollout_correct / HORIZON
        mean_acc      = rollout_acc.mean().item()
        mean_correct  = rollout_correct.mean().item()

        # ---- GRPO advantage & loss -----------------------------------
        base = r_tok.mean(1, keepdim=True)
        adv  = (r_tok - base)
        adv  = (adv - adv.mean()) / (adv.std() + 1e-8)

        flat_gen  = gen.reshape(-1, HORIZON)
        flat_adv  = adv.reshape(-1, HORIZON)

        logp_new = torch.distributions.Categorical(logits=actor(flat_gen)).log_prob(flat_gen)
        logp_ref = torch.distributions.Categorical(logits=ref(flat_gen)).log_prob(flat_gen)
        kl       = (logp_new - logp_ref).mean()

        kl_coef  = min(BETA_KL, BETA_KL * chars_seen / max(1, KL_WARM))
        loss     = -(flat_adv * logp_new).mean() + kl_coef * kl

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        opt.step()

        chars_seen += BATCH * HORIZON

        # ---- logging -------------------------------------------------
        if it % 10 == 0:
            wandb.log({
                "reward_tok":      r_tok.mean().item(),
                "policy_loss":     (-(flat_adv * logp_new).mean()).item(),
                "kl":              kl.item(),
                "kl_coef":         kl_coef,
                "chars":           chars_seen,
                "rollout_acc":     mean_acc,
                "rollout_correct": mean_correct,
            }, step=chars_seen)

        if it % 200 == 0:
            sample = DEC(generate(actor, ctx0[:1].clone(), 120)[0])
            wandb.log({"sample": wandb.Html(f"<pre>{sample}</pre>")}, step=chars_seen)

    run.finish()

# ── local entry ──────────────────────────────────────────────────────────
@stub.local_entrypoint()
def main():
    train_remote.remote()
