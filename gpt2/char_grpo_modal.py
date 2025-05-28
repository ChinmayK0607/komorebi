#!/usr/bin/env python
"""
GRPO on tiny-Shakespeare with a frozen bigram reference.
Now logs per-rollout accuracy (#chars correct / horizon).

Launch:  modal run grpo_bigram_ref.py
"""

# ─── hyper-params ────────────────────────────────────────────────────────
HORIZON, BATCH, K_SAMPLES = 2, 64, 65
TOTAL_ITERS, LR           = 1000, 1e-5
BETA_KL, KL_WARM          = 1e-3, 2000
NEG_REWARD, PARTIAL_LP    = False, 0.0
GPU_TYPE                  = "A10G"          # "A100", "L40S", …

# ─── Modal plumbing ──────────────────────────────────────────────────────
import math, requests, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import modal, wandb, numpy as np, copy

stub  = modal.App("grpo-char-bigram")
image = (
    modal.Image.debian_slim()
    .pip_install("numpy", "torch==2.3.0", "tqdm", "wandb", "requests")
)

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)

def ensure_dataset(p="input.txt"):
    if not Path(p).exists():
        Path(p).write_text(requests.get(DATA_URL, timeout=10).text, encoding="utf-8")

def build_vocab(p="input.txt"):
    text = Path(p).read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi  = {ch: i for i, ch in enumerate(chars)}
    itos  = {i: ch for ch, i in stoi.items()}
    def enc(s): return torch.tensor([stoi[c] for c in s], dtype=torch.long)
    def dec(t): return "".join(itos[int(i)] for i in t)
    return enc, dec, len(chars), stoi, itos, text

class TextLoader:
    def __init__(self, text, enc, B, T):
        self.data = enc(text)
        self.B, self.T, self.pos = B, T, 0
    def next(self):
        span = self.B * self.T + 1
        if self.pos + span > len(self.data):
            self.pos = 0
        buf = self.data[self.pos : self.pos + span]; self.pos += span
        return buf[:-1].view(self.B, self.T), buf[1:].view(self.B, self.T)

# ─── Actor model (tiny GPT) ──────────────────────────────────────────────
class TinyGPT(nn.Module):
    def __init__(self, V, n_layer=6, n_head=6, n_emb=384, blk=256):
        super().__init__()
        self.wte  = nn.Embedding(V, n_emb)
        self.wpe  = nn.Embedding(blk, n_emb)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                n_emb, n_head, 4 * n_emb,
                activation="gelu", batch_first=True
            ) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, V, bias=False)
        self.head.weight = self.wte.weight
    def forward(self, idx):
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for l in self.layers:
            x = l(x)
        return self.head(self.ln_f(x))

# ─── Bigram reference (log-probs) ────────────────────────────────────────
class BigramRef(nn.Module):
    def __init__(self, counts):                     # counts: (V,V)
        super().__init__()
        probs = (counts + 1) / (counts.sum(1, keepdims=True) + counts.size(1))
        self.register_buffer("logits", torch.log(probs))   # (V,V)
    def forward(self, idx):                         # idx: (B,T)
        return self.logits[idx]                     # (B,T,V)

# ─── helpers ─────────────────────────────────────────────────────────────
def tok_reward(gen, ref, neg=False):
    m = (gen == ref).float()
    return m * 2 - 1 if neg else m

def add_partial(r, lp, c): return r if c == 0 else r + c * lp.detach()

@torch.no_grad()
def generate(model, ctx, steps):
    for _ in range(steps):
        nxt = torch.multinomial(torch.softmax(model(ctx)[:, -1], -1), 1)
        ctx = torch.cat([ctx, nxt], 1)
    return ctx[:, -steps:]          # (B,steps)

# ─── training loop ───────────────────────────────────────────────────────
@stub.function(
    gpu=GPU_TYPE,
    image=image,
    timeout=60 * 60 * 3,
    secrets=[modal.Secret.from_name("wandb")]
)
def train_remote():
    torch.set_float32_matmul_precision("high")
    DEV = "cuda" if torch.cuda.is_available() else "cpu"

    # data & vocab
    ensure_dataset()
    ENC, DEC, V, stoi, itos, text = build_vocab()
    loader = TextLoader(text, ENC, BATCH, HORIZON + 1)

    # reference: bigram counts
    counts = torch.zeros(V, V, dtype=torch.float32)
    for a, b in zip(text, text[1:]):
        counts[stoi[a], stoi[b]] += 1
    ref = BigramRef(counts).to(DEV).eval()

    # actor
    actor = TinyGPT(V).to(DEV)
    actor = torch.compile(actor)
    opt   = torch.optim.AdamW(actor.parameters(), lr=LR)

    # wandb
    run = wandb.init(
        project="gpt2-grpo-char",
        config=dict(
            horizon=HORIZON,
            beta=BETA_KL,
            ref="bigram",
            batch=BATCH,
            k_samples=K_SAMPLES,
            total_iters=TOTAL_ITERS,
        ),
    )

    chars_seen = 0
    for it in tqdm(range(1, TOTAL_ITERS + 1), leave=False, desc="iters"):
        ctx0, ref_tok = loader.next()                    # (B,1)  (B,T)
        ctx0, ref_tok = ctx0[:, :1].to(DEV), ref_tok[:, :HORIZON].to(DEV)

        G_list, R_list = [], []
        for _ in range(K_SAMPLES):
            gen = generate(actor, ctx0.clone(), HORIZON)           # (B,T)
            logits = actor(gen)
            dist   = torch.distributions.Categorical(logits.softmax(-1))
            logp   = dist.log_prob(gen)

            r_tok  = tok_reward(gen, ref_tok, NEG_REWARD)
            r_tok  = add_partial(r_tok, logp, PARTIAL_LP)

            G_list.append(gen)      # (B,T)
            R_list.append(r_tok)    # (B,T)

        G = torch.stack(G_list, 1)  # (B,K,T)
        R = torch.stack(R_list, 1)  # (B,K,T)

        # ----- NEW: rollout-level correctness metrics ------------------
        correct_mask = (G == ref_tok.unsqueeze(1))            # (B,K,T)
        rollout_correct = correct_mask.sum(-1).float()        # (B,K)
        rollout_acc     = rollout_correct / HORIZON           # (B,K)
        mean_acc        = rollout_acc.mean().item()
        mean_correct    = rollout_correct.mean().item()

        # ----- GRPO advantage & loss -----------------------------------
        base = R.mean(dim=1, keepdim=True)
        adv  = (R - base)
        adv  = (adv - adv.mean()) / (adv.std() + 1e-8)

        flatG  = G.reshape(-1, HORIZON)
        flatA  = adv.reshape(-1, HORIZON)

        logp_new = torch.distributions.Categorical(logits=actor(flatG)).log_prob(flatG)
        logp_ref = torch.distributions.Categorical(logits=ref(flatG)).log_prob(flatG)
        kl       = (logp_new - logp_ref).mean()

        kl_coef = min(BETA_KL, BETA_KL * chars_seen / max(1, KL_WARM))
        loss = -(flatA * logp_new).mean() + kl_coef * kl

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        opt.step()

        chars_seen += BATCH * HORIZON

        # ----- logging -------------------------------------------------
        if it % 10 == 0:
            wandb.log({
                "reward_tok":    R.mean().item(),
                "policy_loss":   (-(flatA * logp_new).mean()).item(),
                "kl":            kl.item(),
                "kl_coef":       kl_coef,
                "chars":         chars_seen,
                "rollout_acc":   mean_acc,        # NEW
                "rollout_correct": mean_correct,  # NEW
            }, step=chars_seen)

        if it % 200 == 0:
            sample = DEC(generate(actor, ctx0[:1], 120)[0])
            wandb.log({"sample": wandb.Html(f"<pre>{sample}</pre>")}, step=chars_seen)

    run.finish()

# ─── local entry ─────────────────────────────────────────────────────────
@stub.local_entrypoint()
def main():
    train_remote.remote()
