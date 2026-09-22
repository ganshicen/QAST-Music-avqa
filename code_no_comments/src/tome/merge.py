import math
from typing import Callable, Tuple
import torch

def do_nothing(x, mode=None):
    return x

def bipartite_soft_matching(metric: torch.Tensor, r: int, class_token: bool=False, distill_token: bool=False) -> Tuple[Callable, Callable]:
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)
    if r <= 0:
        return (do_nothing, do_nothing)
    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        (a, b) = (metric[..., ::2, :], metric[..., 1::2, :])
        scores = a @ b.transpose(-1, -2)
        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf
        (node_max, node_idx) = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        unm_idx = edge_idx[..., r:, :]
        src_idx = edge_idx[..., :r, :]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode='mean') -> torch.Tensor:
        (src, dst) = (x[..., ::2, :], x[..., 1::2, :])
        (n, t1, c) = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        (unm, dst) = (x[..., :unm_len, :], x[..., unm_len:, :])
        (n, _, c) = unm.shape
        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))
        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)
        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)
        return out
    return (merge, unmerge)

def kth_bipartite_soft_matching(metric: torch.Tensor, k: int) -> Tuple[Callable, Callable]:
    if k <= 1:
        return (do_nothing, do_nothing)

    def split(x):
        t_rnd = x.shape[1] // k * k
        x = x[:, :t_rnd, :].view(x.shape[0], -1, k, x.shape[2])
        (a, b) = (x[:, :, :k - 1, :].contiguous().view(x.shape[0], -1, x.shape[-1]), x[:, :, k - 1, :])
        return (a, b)
    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        (a, b) = split(metric)
        r = a.shape[1]
        scores = a @ b.transpose(-1, -2)
        (_, dst_idx) = scores.max(dim=-1)
        dst_idx = dst_idx[..., None]

    def merge(x: torch.Tensor, mode='mean') -> torch.Tensor:
        (src, dst) = split(x)
        (n, _, c) = src.shape
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        return dst

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        (n, _, c) = x.shape
        dst = x
        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c)).to(x.dtype)
        src = src.view(n, -1, k - 1, c)
        dst = dst.view(n, -1, 1, c)
        out = torch.cat([src, dst], dim=-2)
        out = out.contiguous().view(n, -1, c)
        return out
    return (merge, unmerge)

def random_bipartite_soft_matching(metric: torch.Tensor, r: int) -> Tuple[Callable, Callable]:
    if r <= 0:
        return (do_nothing, do_nothing)
    with torch.no_grad():
        (B, N, _) = metric.shape
        rand_idx = torch.rand(B, N, 1, device=metric.device).argsort(dim=1)
        a_idx = rand_idx[:, :r, :]
        b_idx = rand_idx[:, r:, :]

        def split(x):
            C = x.shape[-1]
            a = x.gather(dim=1, index=a_idx.expand(B, r, C))
            b = x.gather(dim=1, index=b_idx.expand(B, N - r, C))
            return (a, b)
        metric = metric / metric.norm(dim=-1, keepdim=True)
        (a, b) = split(metric)
        scores = a @ b.transpose(-1, -2)
        (_, dst_idx) = scores.max(dim=-1)
        dst_idx = dst_idx[..., None]

    def merge(x: torch.Tensor, mode='mean') -> torch.Tensor:
        (src, dst) = split(x)
        C = src.shape[-1]
        dst = dst.scatter_reduce(-2, dst_idx.expand(B, r, C), src, reduce=mode)
        return dst

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        C = x.shape[-1]
        dst = x
        src = dst.gather(dim=-2, index=dst_idx.expand(B, r, C))
        out = torch.zeros(B, N, C, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=a_idx.expand(B, r, C), src=src)
        out.scatter_(dim=-2, index=b_idx.expand(B, N - r, C), src=dst)
        return out
    return (merge, unmerge)

def merge_wavg(merge: Callable, x: torch.Tensor, size: torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor]:
    if size is None:
        size = torch.ones_like(x[..., 0, None])
    x = merge(x * size, mode='sum')
    size = merge(size, mode='sum')
    x = x / size
    return (x, size)

def merge_source(merge: Callable, x: torch.Tensor, source: torch.Tensor=None) -> torch.Tensor:
    if source is None:
        (n, t, _) = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)
    source = merge(source, mode='amax')
    return source
