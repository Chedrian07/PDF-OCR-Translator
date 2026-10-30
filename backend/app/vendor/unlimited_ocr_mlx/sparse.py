"""Unlimited-OCR-2 SR-SWA inference, ported from the pinned Baidu DASH code.

Page state and KV live in a request-owned cache. Routing uses head_cls summaries,
top-16 entmax, 12 remote blocks and the 128-token local window. Main and router
RoPE both use the trained page2d coordinates. No CUDA/Triton dependency.
"""

from __future__ import annotations

import math
import numpy as np
import mlx.core as mx
import mlx.nn as nn


class SparseCache:
    def __init__(self, config):
        self.config = config
        self.keys = self.values = self.index_keys = None
        self.offset = 0
        self.page = self.tok = 0
        self.counts = None
        self.image_pages = None
        self.marker_count = self.prompt_pages = 0
        self.coords = self.candidates = None

    def prepare(self, token_ids):
        ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        pg_id = self.config["page_token_id"]
        img_id = self.config["image_token_id"]
        bs = self.config["sparse_block_size"]
        if self.offset == 0:
            img = ids == img_id
            if not img.any() or not (ids == pg_id).any():
                raise ValueError(
                    "Unlimited-OCR-2 requires <PAGE><image> prompt markers"
                )
            cum = np.cumsum(ids == pg_id)
            last_img = np.flatnonzero(img)[-1]
            self.prompt_pages = int(cum[last_img])
            self.marker_count = int(cum[-1])
            pages = np.maximum(
                np.where(np.arange(len(ids)) <= last_img, cum, cum - self.prompt_pages),
                0,
            )
            counts = np.zeros(self.prompt_pages + 1, dtype=np.int64)
            tok = []
            for page in pages:
                tok.append(int(counts[page]))
                counts[page] += 1
            self.counts = np.bincount(pages[: last_img + 1], minlength=len(counts))
            self.page, self.tok = int(pages[-1]), int(tok[-1])
            self.image_pages = np.where(img, cum - 1, -1)
            query_pages = np.where(
                np.arange(len(ids)) <= last_img, cum - 1, cum - 1 - self.prompt_pages
            )
        else:
            if len(ids) != 1:
                raise ValueError("SparseCache supports one decode token per step")
            if ids[0] == pg_id:
                self.marker_count += 1
                self.page = min(self.page + 1, len(self.counts) - 1)
                self.tok = int(self.counts[self.page])
            else:
                self.tok += 1
            pages, tok = [self.page], [self.tok]
            query_pages = [self.marker_count - 1 - self.prompt_pages]
        self.coords = (
            mx.array(np.asarray(tok)[None], dtype=mx.int32),
            mx.array(np.asarray(pages)[None], dtype=mx.int32),
        )
        nblk = (self.offset + len(ids) + bs - 1) // bs
        pad = np.full(nblk * bs, -1, dtype=np.int64)
        pad[: len(self.image_pages)] = self.image_pages
        blocks = pad.reshape(nblk, bs)
        candidate = np.array(
            [
                np.any(blocks == q, axis=-1) if q >= 0 else np.ones(nblk, bool)
                for q in query_pages
            ]
        )
        candidate = np.where(candidate.any(axis=-1, keepdims=True), candidate, True)
        self.candidates = mx.array(candidate[None])

    def update(self, k, v, index_k):
        self.keys = k if self.keys is None else mx.concatenate((self.keys, k), axis=2)
        self.values = (
            v if self.values is None else mx.concatenate((self.values, v), axis=2)
        )
        self.index_keys = (
            index_k
            if self.index_keys is None
            else mx.concatenate((self.index_keys, index_k), axis=1)
        )
        self.offset += k.shape[2]


def rope2d(x, coords, sp):
    half = x.shape[-1] // 2
    npg = sp["sparse_rope_page_pairs"]
    ntok = half - npg
    ti = sp["sparse_rope_tok_theta"] ** (-mx.arange(ntok, dtype=mx.float32) / ntok)
    pi = sp["sparse_rope_page_theta"] ** (-mx.arange(npg, dtype=mx.float32) / npg)
    tok, page = coords
    mod = sp.get("sparse_rope_page_mod", 0)
    if mod:
        page = page % mod
    ang = mx.concatenate(
        (
            tok[..., None].astype(mx.float32) * ti,
            page[..., None].astype(mx.float32) * pi,
        ),
        axis=-1,
    )
    emb = mx.concatenate((ang, ang), axis=-1)[:, None]
    cos, sin = mx.cos(emb).astype(x.dtype), mx.sin(emb).astype(x.dtype)
    rot = mx.concatenate((-x[..., half:], x[..., :half]), axis=-1)
    return x * cos + rot * sin


def entmax15(x):
    x = (x - mx.max(x, axis=-1, keepdims=True)) / 2
    xs = -mx.sort(-x, axis=-1)
    rho = mx.arange(1, x.shape[-1] + 1, dtype=mx.float32)
    mean = mx.cumsum(xs, axis=-1) / rho
    mean_sq = mx.cumsum(xs * xs, axis=-1) / rho
    delta = (1 - rho * (mean_sq - mean * mean)) / rho
    taus = mean - mx.sqrt(mx.maximum(delta, 0))
    support = mx.sum(taus <= xs, axis=-1, keepdims=True)
    tau = mx.take_along_axis(taus, mx.maximum(support - 1, 0).astype(mx.int32), axis=-1)
    return mx.maximum(x - tau, 0) ** 2


class RouterNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = mx.zeros((dim,))
        self.eps = eps

    def __call__(self, x):
        y = mx.nan_to_num(x.astype(mx.float32), nan=0, posinf=0, neginf=0)
        y = y * mx.rsqrt(mx.mean(y * y, axis=-1, keepdims=True) + self.eps)
        y = mx.clip(y, -1e4, 1e4)
        return mx.clip(y * (1 + self.weight.astype(mx.float32)), -1e4, 1e4).astype(
            x.dtype
        )


class Router(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.sparse_attention_config["sparse_index_dim"]
        self.q_proj = nn.Linear(config.hidden_size, d, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, d, bias=False)
        self.q_norm = RouterNorm(d, eps=config.rms_norm_eps)
        self.k_norm = RouterNorm(d, eps=config.rms_norm_eps)
        self.head_cls = mx.zeros((d,))

    def qk(self, x, coords, sp):
        # Upstream disables autocast for the router but retains checkpoint dtype.
        x = mx.nan_to_num(x, nan=0, posinf=0, neginf=0)
        q = self.q_norm(self.q_proj(x))[:, None]
        k = self.k_norm(self.k_proj(x))[:, None]
        return (
            mx.nan_to_num(
                rope2d(q, coords, sp)[:, 0].astype(mx.float32),
                nan=0,
                posinf=0,
                neginf=0,
            ),
            mx.nan_to_num(
                rope2d(k, coords, sp)[:, 0].astype(mx.float32),
                nan=0,
                posinf=0,
                neginf=0,
            ),
        )


class SparseAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.sp = sp = config.sparse_attention_config
        self.layer_idx = layer_idx
        self.n_heads, self.n_kv_heads = (
            config.num_attention_heads,
            config.num_key_value_heads,
        )
        self.head_dim = config.head_dim
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(
                self,
                name,
                nn.Linear(
                    config.hidden_size, config.hidden_size, bias=config.attention_bias
                ),
            )
        self.dash_router = Router(config)

    def __call__(self, x, mask=None, cache=None):
        if cache is None or cache.coords is None:
            raise ValueError("SR-SWA requires a prepared request cache")
        B, L, D = x.shape
        sp, router = self.sp, self.dash_router
        bs, local_blocks = sp["sparse_block_size"], sp["sparse_local_block"]
        local_window = bs * local_blocks

        def shape(a):
            return a.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = rope2d(shape(self.q_proj(x)), cache.coords, sp)
        k = rope2d(shape(self.k_proj(x)), cache.coords, sp)
        v = shape(self.v_proj(x))
        iq, ik = router.qk(x, cache.coords, sp)
        offset = cache.offset
        cache.update(k, v, ik)
        S = cache.offset
        nblk = (S + bs - 1) // bs
        pad = nblk * bs - S
        kk = mx.pad(cache.index_keys, ((0, 0), (0, pad), (0, 0))).reshape(
            B, nblk, bs, -1
        )
        real = (mx.arange(nblk * bs) < S).reshape(nblk, bs)
        d = kk.shape[-1]
        aw = mx.clip(
            mx.sum(kk * router.head_cls.astype(mx.float32), axis=-1) / math.sqrt(d),
            -1e4,
            1e4,
        )
        aw = mx.where(real, aw, -1e30)
        weights = mx.softmax(aw, axis=-1)
        summary = mx.sum(weights[..., None] * kk, axis=2)
        scale = sp["dash_scale_per_layer"][self.layer_idx] / math.sqrt(d)
        floor = sp["dash_floor_per_layer"][self.layer_idx]
        results = []
        # Bound temporary gathered tensors, without reducing input resolution or read set.
        for start in range(0, L, 32):
            end = min(L, start + 32)
            pos = mx.arange(offset + start, offset + end)
            score = mx.clip(
                iq[:, start:end] @ summary.transpose(0, 2, 1) / math.sqrt(d), -1e4, 1e4
            )
            remote = mx.arange(nblk)[None, None] <= (
                pos[None, :, None] // bs - local_blocks
            )
            cand = remote & cache.candidates[:, start:end]
            cand = mx.where(mx.any(cand, axis=-1, keepdims=True), cand, remote)
            score = mx.where(cand, score * scale, -1e4)
            M = min(16, nblk)
            topi = mx.argsort(-score, axis=-1)[..., :M]
            prior = entmax15(mx.take_along_axis(score, topi, axis=-1))
            K = min(sp["dash_topk_blocks"], M)
            topi, prior = topi[..., :K], prior[..., :K]
            valid = mx.take_along_axis(cand, topi, axis=-1)
            logp = mx.log(mx.maximum(prior, floor))
            mu = mx.sum(mx.where(valid, logp, 0), axis=-1, keepdims=True) / mx.maximum(
                mx.sum(valid, axis=-1, keepdims=True), 1
            )
            bias = (logp - mu) / sp["dash_sigma"]
            # Gather the selected remote blocks and local window, excluding overlap.
            remote_ids = (topi[..., None] * bs + mx.arange(bs)).reshape(
                B, end - start, -1
            )
            remote_valid = mx.repeat(valid, bs, axis=-1) & (remote_ids < S)
            remote_valid &= remote_ids <= pos[None, :, None] - local_window
            local_ids = mx.broadcast_to(
                pos[None, :, None] - mx.arange(local_window - 1, -1, -1),
                (B, end - start, local_window),
            )
            ids = mx.concatenate((remote_ids, local_ids), axis=-1)
            valid_tokens = mx.concatenate((remote_valid, local_ids >= 0), axis=-1)
            bias_tokens = mx.concatenate(
                (mx.repeat(bias, bs, axis=-1), mx.zeros(local_ids.shape)), axis=-1
            )
            safe_ids = mx.clip(ids, 0, S - 1)
            # B=1 is the product's request contract.
            keys = mx.take(cache.keys[0], safe_ids[0], axis=1).transpose(1, 0, 2, 3)
            vals = mx.take(cache.values[0], safe_ids[0], axis=1).transpose(1, 0, 2, 3)
            queries = q[0, :, start:end, :].transpose(1, 0, 2)[:, :, None, :]
            logits = mx.sum(
                queries.astype(mx.float32) * keys.astype(mx.float32), axis=-1
            ) / math.sqrt(self.head_dim)
            logits += bias_tokens[0, :, None, :]
            logits = mx.where(valid_tokens[0, :, None, :], logits, -1e30)
            probs = mx.softmax(logits, axis=-1)
            out = mx.sum(probs[..., None] * vals.astype(mx.float32), axis=2)
            results.append(out.transpose(1, 0, 2)[None].astype(x.dtype))
        output = mx.concatenate(results, axis=2).transpose(0, 2, 1, 3).reshape(B, L, D)
        return self.o_proj(output)
