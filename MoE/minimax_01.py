import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

@dataclass
class config: 
    num_experts: int = 32
    n_layer : int = 80
    n_head : int = 64
    head_dim : int = 128
    d_model : int = 6144
    rope_theta : float = 10000.00
    expert_topk : int = 2
    n_kv_heads : int = 8
    block_size : int = 1024 # not mentioned in paper

# RMS Norm
def norm(x):
    """RMSNorm implementation - compatible with PyTorch 2.3"""
    # RMS normalization: x / sqrt(mean(x^2) + eps)
    return F.rms_norm(x, (x.size(-1),))

def precompute_theta_pos_frequencies(
    head_dim: int, seq_len: int, device: str, theta: float = 10000.0
):
    """
    Precompute the complex exponent for rotary embeddings.
    """
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
    theta_numerator = torch.arange(0, head_dim, 2).float()
    base = theta ** (theta_numerator / head_dim)
    theta_inv = 1.0 / base
    theta_inv = theta_inv.to(device)
    m = torch.arange(seq_len, device=device)
    freqs = torch.outer(m, theta_inv).float()
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    """
    Apply the precomputed rotary embeddings to the Q or K vector x.
    """
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)  # (1, Seq_Len, 1, Head_Dim/2)
    x_rotated = x_complex * freqs_complex
    x_out = torch.view_as_real(x_rotated)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

class SoftMaxAttn(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_kv_heads = config.n_kv_heads
        self.q_heads = config.n_head
        self.n_rep = config.n_head // config.n_kv_heads
        self.q = nn.Linear(config.n_head, config.head_dim * config.n_head,bias = False)
        self.k= nn.Linear(config.n_head, config.head_dim * config.n_kv_heads,bias = False)
        self.v = nn.Linear(config.n_head, config.head_dim * config.n_kv_heads,bias = False)
        self.o_proj = nn.Linear(config.n_head, config.head_dim * config.n_head,bias = False)
        self.register_buffer('bias',torch.tril(torch.ones(config.block_size,config.block_size)).view(1,1,config.block_size,config.block_size))
    def forward(self,x,freqs_complex):
        B,T,C = x.size()

        query, key, value = self.q(x),self.k(x),self.v(x)
        query = query.view(B,T,self.n_head, self.head_dim) # B,T,n_h, head_dim
        key = key.view(B,T,self.n_kv_heads,self.head_dim)
        value = value.view(B,T,self.n_kv_heads, self.head_dim)

        query, key = apply_rotary_embeddings(query,freqs_complex,device = x.device), apply_rotary_embeddings(key,freqs_complex, device = x.device)

        q,k,v = query.transpose(1,2), key.transpose(1,2), value.transpose(1,2)
        
        attn_output = F.scaled_dot_product_attention(q,k,v,is_causal = True)
        attn_output = attn_output.transpose(1,2).contiguous.view(B,T,C)
        return self.o_proj(attn_output)

class _LAKernel(nn.Module):
    def __init__(self, bs):
        super().__init__()
        self.bs = bs
        self.register_buffer("mask", torch.tril(torch.ones(bs, bs)))
    def forward(self, q, k, v):
        *b,N,d = q.shape; T = N // self.bs; bs = self.bs
        q,k,v = [x.view(*b,T,bs,d) for x in (q,k,v)]
        kv = torch.zeros(*b,d,d, device=q.device, dtype=q.dtype)
        o  = torch.empty_like(q).reshape(*b,N,d)
        for t in range(T):
            qt,kt,vt = q[...,t,:,:],k[...,t,:,:],v[...,t,:,:]
            o[...,t*bs:(t+1)*bs,:] = ((qt @ kt.transpose(-2,-1))*self.mask)@vt + qt@kv
            kv += kt.transpose(-2,-1) @ vt
        return o

class LightningBlock(nn.Module):
    def __init__(self, cfg: config):
        super().__init__()
        d, nh, hd = cfg.d_model, cfg.n_head, cfg.head_dim
        self.q = nn.Linear(d, d, False)
        self.k = nn.Linear(d, nh//cfg.n_kv_heads*cfg.n_kv_heads*hd, False)
        self.v = nn.Linear(d, nh//cfg.n_kv_heads*cfg.n_kv_heads*hd, False)
        self.g = nn.Linear(d, d, False)
        self.act, self.sig = nn.SiLU(), nn.Sigmoid()
        self.kernel = _LAKernel(cfg.block_size)
        self.proj   = nn.Linear(d, d, False)
        self.norm   = norm(d)
    def forward(self, x):
        x = norm(x)
        q = self.act(self.q(x)); k = self.act(self.k(x)); v = self.act(self.v(x))
        g = self.sig(self.g(x))
        o = self.kernel(q,k,v) * g
        return self.proj(o)

class Expert(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ffn = nn.Linear(config.d_model, 1.5 * config.d_model) # (6144 , 9216)
        self.gelu = nn.GELU(approximate='tanh') # using as no information in paper about activation
        self.ffn_proj = nn.Linear(1.5 * config.d_model, config.d_model) #
    def forward(self,x):
        x = self.ffn(x)
        # activation
        x = self.gelu(x)
        x = self.ffn_proj(x)

        return x


class MoELayer(nn.Module):
    """Top-K router + experts with auxiliary load-balancing loss."""
    def __init__(self, cfg: config, alpha_aux: float = 0.02):
        super().__init__()
        self.cfg       = cfg
        self.alpha_aux = alpha_aux
        self.router    = nn.Linear(cfg.d_model, cfg.num_experts, bias=False)
        self.experts   = nn.ModuleList([Expert(cfg) for _ in range(cfg.num_experts)])

    def forward(self, x):
        B, T, D  = x.size()
        N        = B * T
        x_flat   = x.view(N, D)

        # ---------- gating ----------
        logits, idx = torch.topk(self.router(x_flat), self.cfg.expert_topk, dim=-1)          # (N , k)
        gate        = torch.softmax(logits, dim=-1)                                          # (N , k)

        # dense (N , E) matrix with routing probs
        dispatch = torch.zeros(N, self.cfg.num_experts, device=x.device, dtype=x.dtype)
        dispatch.scatter_(1, idx, gate)

        # ---------- experts ----------
        out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            sel = dispatch[:, i]                                                             # (N,)
            if sel.nonzero().numel() == 0:       # no tokens for this expert
                continue
            tok_idx      = sel.nonzero(as_tuple=True)[0]
            expert_out   = expert(x_flat.index_select(0, tok_idx))                          # (n_i , D)
            out.index_add_(0, tok_idx, expert_out * sel[tok_idx].unsqueeze(-1))             # weighted add

        out = out.view(B, T, D)

        # ---------- auxiliary loss ----------
        tok_per_expert = (dispatch > 0).sum(0).float()                                       # (E,)
        f_i            = tok_per_expert / tok_per_expert.sum()                               # fractions
        m_i            = dispatch.sum(0) / tok_per_expert.clamp(min=1.)                      # mean prob
        aux_loss       = self.alpha_aux * (f_i * m_i).mean()

        return out, aux_loss


class LightningMoeBlock(nn.Module):
    def __init__(self, cfg: config):
        super().__init__()
        self.attn = LightningBlock(cfg)
        self.moe  = MoELayer(cfg)
    def forward(self, x):
        h = x + self.attn(x)
        h_norm = norm(h)
        y, aux = self.moe(h_norm)
        return h + y, aux

class _SoftmaxMHA(nn.Module):
    def __init__(self, cfg: config):
        super().__init__()
        self.n_head  = cfg.n_head
        self.h_dim   = cfg.head_dim
        proj = cfg.n_head * cfg.head_dim
        self.q = nn.Linear(cfg.d_model, proj, False)
        self.k = nn.Linear(cfg.d_model, proj, False)
        self.v = nn.Linear(cfg.d_model, proj, False)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, False)
    def forward(self, x, freqs):
        B,T,_ = x.size()
        q = self.q(x).view(B,T,self.n_head,self.h_dim)
        k = self.k(x).view(B,T,self.n_head,self.h_dim)
        v = self.v(x).view(B,T,self.n_head,self.h_dim)
        q,k = apply_rotary_embeddings(q,freqs,x.device), apply_rotary_embeddings(k,freqs,x.device)
        q,k,v = [t.transpose(1,2) for t in (q,k,v)]
        y = F.scaled_dot_product_attention(q,k,v,is_causal=True)
        y = y.transpose(1,2).contiguous().view(B,T,-1)
        return self.o(y)

class SoftmaxMoeBlock(nn.Module):
    def __init__(self, cfg: config, seq_len: int):
        super().__init__()
        self.freqs = precompute_theta_pos_frequencies(cfg.head_dim, seq_len, 'cpu', cfg.rope_theta)
        self.attn  = _SoftmaxMHA(cfg)
        self.moe   = MoELayer(cfg)
    def forward(self, x):
        h = x + self.attn(norm(x), self.freqs[:x.size(1)].to(x.device))
        h_norm = norm(h)
        y, aux = self.moe(h_norm)
        return h + y, aux

class LightningMoEModel(nn.Module):
    def __init__(self, cfg: config, seq_len: int = 128):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(cfg.n_layer):
            if i % 8 == 7:
                self.layers.append(SoftmaxMoeBlock(cfg, seq_len))
            else:
                self.layers.append(LightningMoeBlock(cfg))
        self.final_norm = norm
    def forward(self, x):
        aux_loss = 0.0
        for layer in self.layers:
            x, aux = layer(x)
            aux_loss = aux_loss + aux
        return self.final_norm(x), aux_loss



