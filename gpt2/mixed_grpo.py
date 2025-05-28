#!/usr/bin/env python
"""
tiny-Shakespeare: 4 k CE warm-up ➜ GRPO with bigram reference
Launch on Modal:   modal run grpo_bigram_ref.py
"""

# ───── hyper-parameters ────────────────────────────────────────────────
PRETRAIN_ITERS  = 4_000        # pure cross-entropy stage
RL_ITERS        = 2_000        # GRPO stage
HORIZON         = 32
BATCH           = 64
K_SAMPLES       = 70
LR              = 3e-5
BETA_KL         = 3e-3
KL_WARM         = 20_000
PARTIAL_LP      = 0.05
NEG_REWARD      = False
GPU_TYPE        = "A10G"       # "A100", …

# ───── Modal plumbing ──────────────────────────────────────────────────
import math, requests, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import modal, wandb, numpy as np

stub = modal.App("grpo-char-bigram-CE")
image = (
    modal.Image.debian_slim()
    .pip_install("numpy", "torch==2.3.0", "tqdm", "wandb", "requests")
)

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def ensure_dataset(f="input.txt"):
    if not Path(f).exists():
        Path(f).write_text(requests.get(DATA_URL, timeout=10).text, encoding="utf-8")

def build_vocab(f="input.txt"):
    txt  = Path(f).read_text(encoding="utf-8")
    ch   = sorted(set(txt))
    stoi = {c:i for i,c in enumerate(ch)}
    itos = {i:c for c,i in stoi.items()}
    enc  = lambda s: torch.tensor([stoi[c] for c in s], dtype=torch.long)
    dec  = lambda t: "".join(itos[int(i)] for i in t)
    return enc, dec, len(ch), stoi, txt

class TextLoader:
    def __init__(self, toks, B, T):
        self.toks, self.B, self.T, self.pos = toks, B, T, 0
    def next(self):
        span = self.B*self.T + 1
        if self.pos + span > len(self.toks): self.pos = 0
        buf = self.toks[self.pos:self.pos+span]; self.pos += span
        return buf[:-1].view(self.B,self.T), buf[1:].view(self.B,self.T)

# ───── tiny GPT actor ──────────────────────────────────────────────────
class TinyGPT(nn.Module):
    def __init__(self, V, n_layer=6, n_head=6, n_emb=384, blk=256):
        super().__init__()
        self.wte, self.wpe = nn.Embedding(V,n_emb), nn.Embedding(blk,n_emb)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                n_emb, n_head, 4*n_emb, activation="gelu", batch_first=True
            ) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, V, bias=False); self.head.weight = self.wte.weight
    def forward(self, idx):
        B,T = idx.shape; pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for l in self.layers: x = l(x)
        return self.head(self.ln_f(x))               # (B,T,V)

# ───── frozen bigram reference ──────────────────────────────────────────
class BigramRef(nn.Module):
    def __init__(self, cnt):
        super().__init__()
        probs = (cnt+1)/(cnt.sum(1,keepdims=True)+cnt.size(1))
        self.register_buffer("logits", torch.log(probs))
    def forward(self, idx): return self.logits[idx]

# ───── helpers ──────────────────────────────────────────────────────────
def tok_reward(g, r, neg=False): m=(g==r).float(); return m*2-1 if neg else m
def add_partial(r, lp): return r + PARTIAL_LP*lp.detach() if PARTIAL_LP else r
@torch.no_grad()
def generate(m, ctx, steps):
    for _ in range(steps):
        nxt = torch.multinomial(torch.softmax(m(ctx)[:,-1],-1),1)
        ctx = torch.cat([ctx,nxt],1)
    return ctx[:,-steps:]

# ───── training ─────────────────────────────────────────────────────────
@stub.function(gpu=GPU_TYPE, image=image, timeout=60*60*3,
               secrets=[modal.Secret.from_name("wandb")])
def train_remote():
    torch.set_float32_matmul_precision("high")
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_dataset()
    ENC, DEC, V, stoi, text = build_vocab()
    loader = TextLoader(ENC(text), BATCH, HORIZON+1)

    # bigram reference
    cnt = torch.zeros(V,V)
    for a,b in zip(text,text[1:]): cnt[stoi[a],stoi[b]] += 1
    ref = BigramRef(cnt).to(DEV).eval()

    actor = TinyGPT(V).to(DEV)
    opt   = torch.optim.AdamW(actor.parameters(), lr=LR)

    run = wandb.init(project="gpt2-grpo-char",
                     config=dict(pretrain=PRETRAIN_ITERS,h=HORIZON,k=K_SAMPLES))

    chars_seen = 0
    total_steps = PRETRAIN_ITERS + RL_ITERS

    for step in tqdm(range(1,total_steps+1),leave=False):
        ctx, tgt = loader.next()
        ctx, tgt = ctx.to(DEV), tgt.to(DEV)

        # ─── Stage 1: cross-entropy warm-up ───────────────────────────
        if step <= PRETRAIN_ITERS:
            logits = actor(ctx)                                # (B,T,V)
            ce_loss = F.cross_entropy(logits.view(-1, V), tgt.flatten())
            opt.zero_grad(set_to_none=True); ce_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(),1.0); opt.step()

            chars_seen += BATCH*HORIZON
            if step % 100 == 0:
                run.log({"stage":"CE","ce_loss":ce_loss.item(),"chars":chars_seen}, step=chars_seen)
            continue   # go to next loop iteration

        # ─── Stage 2: GRPO RL ────────────────────────────────────────
        ctx0 = ctx[:,:1]                       # first char as prompt  (B,1)
        ref_tok = tgt                         # ground-truth horizon  (B,T)

        prompt = ctx0.repeat_interleave(K_SAMPLES,0)          # (B*K,1)
        gen    = generate(actor, prompt, HORIZON)             # (B*K,T)
        logits = actor(gen)                                   # (B*K,T,V)
        dist   = torch.distributions.Categorical(logits.softmax(-1))
        logp   = dist.log_prob(gen)                           # (B*K,T)

        gen   = gen.view(BATCH,K_SAMPLES,HORIZON)
        logp  = logp.view(BATCH,K_SAMPLES,HORIZON)

        r_tok = tok_reward(gen, ref_tok.unsqueeze(1), NEG_REWARD)
        r_tok = add_partial(r_tok, logp)                      # (B,K,T)

        base = r_tok.mean(1,keepdim=True)
        adv  = (r_tok - base)
        adv  = (adv - adv.mean())/(adv.std()+1e-8)

        flatG = gen.flatten(0,1)
        flatA = adv.flatten(0,1)

        logp_new = torch.distributions.Categorical(logits=actor(flatG)).log_prob(flatG)
        logp_ref = torch.distributions.Categorical(logits=ref(flatG)).log_prob(flatG)
        kl = (logp_new - logp_ref).mean()

        kl_coef = min(BETA_KL, BETA_KL*chars_seen/max(1,KL_WARM))
        loss = -(flatA*logp_new).mean() + kl_coef*kl

        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(),1.0); opt.step()

        # accuracy log
        acc = (gen == ref_tok.unsqueeze(1)).float().mean().item()

        chars_seen += BATCH*HORIZON
        if step % 50 == 0:
            run.log({"stage":"RL","loss":loss.item(),"kl":kl.item(),
                     "kl_coef":kl_coef,"rollout_acc":acc,"chars":chars_seen},
                    step=chars_seen)

        if step % 400 == 0:
            sample = DEC(generate(actor, ctx0[:1].clone(), 120)[0])
            run.log({"sample":wandb.Html(f"<pre>{sample}</pre>")}, step=chars_seen)

    run.finish()

@stub.local_entrypoint()
def main():
    train_remote.remote()
