"""
HHR ops from https://arxiv.org/pdf/2109.02157.pdf
"""
import copy

import torch
from torch.distributions import Normal


def fft(x):
    return torch.fft.rfft(x, norm='ortho', dim=-1)


def ifft(x):
    return torch.fft.irfft(x, norm='ortho', dim=-1)


def bind(a, b):
    return ifft(torch.multiply(fft(a), fft(b)))


def unbind(s, a):
    return bind(s, inverse(a))


def inverse(a):
    a = torch.flip(a, dims=[-1])
    return torch.roll(a, 1, dims=-1)


def unit_projection(x):
    c = fft(x)
    c_ish = c / torch.norm(c, dim=-1, keepdim=True)
    output = ifft(c_ish)
    return torch.real(output)


def init(shape):
    a = torch.randn(*shape) / shape[-1]
    return unit_projection(a)


def unit_regularization(v):
    v_hat = fft(v)
    v_hat = v_hat * torch.norm(v_hat, dim=-1, keepdim=True)
    x = torch.real(ifft(v_hat))
    dist = Normal(0., 1. / v.shape[-1])
    nlp = -dist.log_prob(x)
    return nlp


def key_value_query(
    k: torch.Tensor, v: torch.Tensor, q: torch.Tensor,
    causal: bool = True, norm: bool = False
):
    """
    Create a key-value vector and then retrieve queried values using HRR.

    This function is meant to reduce the number of fft/ifft calls compared to naively
    binding k/v, summing over the sequence, and then unbinding q.
    """
    # NOTE: perhaps we can avoid explicitly inverting and assume inv_q is learned by the model?
    # k, v, inv_q = fft(k), fft(v), inverse(fft(q))
    k, v, inv_q = fft(k), fft(v), fft(q)
    kv = k * v
    if causal:
        r = kv.cumsum(dim=-2) #* kv.shape[-1] / kv.shape[-2]
    else:
        r = kv.sum(dim=-2, keepdim=True)
    # unbind values for each query
    qv = torch.real(ifft(r * inv_q))
    return qv


def perm_key_value_query(
    k: torch.Tensor, v: torch.Tensor, q: torch.Tensor, perms: torch.Tensor,
    causal: bool = True,
):
    """
    Create a key-value vector and then retrieve queried values using HRR.

    This function is meant to reduce the number of fft/ifft calls compared to naively
    binding k/v, summing over the sequence, and then unbinding q.
    """
    # NOTE: perhaps we can avoid explicitly inverting and assume inv_q is learned by the model?
    # k, v, inv_q = fft(k), fft(v), inverse(fft(q))
    k, v, inv_q = fft(k), fft(v), fft(q)
    inv_q = inv_q[..., perms].permute(2, 0, 1, 3)
    k = k[..., perms].permute(2, 0, 1, 3)
    v = v[None, ...]
    kv = k * v
    if causal:
        r = kv.cumsum(dim=-2) #* kv.shape[-1] / kv.shape[-2]
    else:
        r = kv.sum(dim=-2, keepdim=True)
    # unbind values for each query/permutation and take the mean
    qv = (r * inv_q).mean(0)
    qv = torch.real(ifft(qv))
    return qv

