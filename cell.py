import torch
import torch.nn as nn
from random import randrange
from torch import einsum
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
import ops


def exists(val):
    return val is not None


def pair(val):
    return (val, val) if not isinstance(val, tuple) else val


def dropout_layers(layers, prob_survival):
    if prob_survival == 1:
        return layers

    num_layers = len(layers)
    to_drop = torch.zeros(num_layers).uniform_(0., 1.) > prob_survival

    # make sure at least one layer makes it
    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False

    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, alphas):
        weights = F.sigmoid(alphas[2])
        return weights[0] * x + weights[1] * self.fn(x, alphas)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, alphas):
        #x = self.norm(x)
        return self.fn(x, alphas)


class Attention(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inner, causal=False):
        super().__init__()
        self.scale = dim_inner ** -0.5
        self.causal = causal

        self.to_qkv = nn.Linear(dim_in, dim_inner * 3, bias=False)
        self.to_out = nn.Linear(dim_inner, dim_out)

    def forward(self, x):
        device = x.device
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if self.causal:
            mask = torch.ones(sim.shape[-2:], device=device).triu(1).bool()
            sim.masked_fill_(mask[None, ...], -torch.finfo(q.dtype).max)

        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        return self.to_out(out)


class SpatialGatingUnit(nn.Module):
    def __init__(
            self,
            max_dim_ff,
            seq_len,
            act=nn.Identity(),
            heads=1,
            init_eps=1e-3,
    ):
        super().__init__()
        dim_out = max_dim_ff // 2
        self.heads = heads
        #self.norm = nn.LayerNorm(dim_out)

        self.act = act

        shape = (heads, seq_len, seq_len)
        weight = torch.zeros(shape)

        self.weight = nn.Parameter(weight)
        init_eps /= seq_len
        nn.init.uniform_(self.weight, -init_eps, init_eps)

        self.bias = nn.Parameter(torch.ones(heads, seq_len))

    def forward(self, x, gate_res=None):
        device, n, h = x.device, x.shape[1], self.heads

        res, gate = x.chunk(2, dim=-1)
        #gate = self.norm(gate)

        weight, bias = self.weight, self.bias

        gate = rearrange(gate, 'b n (h d) -> b h n d', h=h)

        gate = einsum('b h n d, h m n -> b h m d', gate, weight)
        gate = gate + rearrange(bias, 'h n -> () h n ()')

        gate = rearrange(gate, 'b h n d -> b n (h d)')

        if exists(gate_res):
            gate = gate + gate_res

        return self.act(gate) * res


class gMLPBlock(nn.Module):
    def __init__(
            self,
            dims,
            dims_ff,
            seq_len,
            heads=1,
            attn_dim=None,
            causal=False,
            act=nn.Identity(),
            circulant_matrix=False
    ):
        super().__init__()

        self.proj_in = ops.ProjMixedOp(in_dim=max(dims), out_dim=max(dims_ff), seq_len=seq_len, bias=True, out_dims=dims_ff)
        self.GELU = nn.GELU()
        self.sgu = SpatialGatingUnit(max_dim_ff=max(dims_ff), seq_len=seq_len)
        self.proj_out = ops.ProjMixedOp(in_dim=max(dims_ff) // 2, out_dim=max(dims), seq_len=seq_len, bias=True, out_dims=dims)

    def forward(self, x, alphas):
        x = self.proj_in(x, alphas[0])
        x = self.GELU(x)
        x = self.sgu(x)  # this is not learnable, W is (n x n)
        x = self.proj_out(x, alphas[1])
        return x

class SearchCellgMLP(nn.Module):
    """ Cell for search
    Each edge is mixed and continuous relaxed.
    """
    def __init__(self, dims, dims_ff, num_patches):
        """
        Args:
            n_nodes: # of intermediate n_nodes
            max_hidden_size: size of feature maps in every node
            bias: whether the linar projections use bias or not
            pre0: if s0 should be preprocessed (n_nodes * hs  ->  hs) or not
            pre1: if s0 should be preprocessed (n_nodes * hs  ->  hs) or not
        """
        super().__init__()
        self.pipeline = Residual(
            PreNorm(  # is now deactivated
                max(dims),  # ojo con la norm y las dimensions
                gMLPBlock(
                    dims=dims,
                    dims_ff=dims_ff,
                    seq_len=num_patches
                )
            )
        )

    def forward(self, x, alphas):
        x = self.pipeline(x, alphas)
        return x


class SearchCellMixer(nn.Module):
    def __init__(self, num_patches, hidden_size, hidden_s_candidates, hidden_c_candidates, drop_p, off_act):
        super(SearchCellMixer, self).__init__()
        self.mlp1 = searchMLP1(num_patches, hidden_s_candidates, hidden_size, drop_p, off_act)
        self.mlp2 = searchMLP2(hidden_size, hidden_c_candidates, drop_p, off_act)

    def forward(self, x, alphas):
        z = self.mlp1(x, alphas[0])  # search mixer
        z = self.mlp2(z, alphas[1])  # search mixer
        return z


class searchMLP1(nn.Module):
    def __init__(self, num_patches, hidden_s_candidates, hidden_size, drop_p, off_act):
        super(searchMLP1, self).__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.T = Rearrange('b s c -> b c s')  # Transpose token and channel axis only
        self.mixed_op = ops.mixedInverseAutoencoder(num_patches, hidden_s_candidates, drop_p, off_act)

    def forward(self, x, alphas):
        z = self.ln(x)
        z = self.T(z)
        z = self.mixed_op(z, alphas)
        z = self.T(z)
        return z + x


class searchMLP2(nn.Module):
    def __init__(self, hidden_size, hidden_c_candidates, drop_p, off_act):
        super(searchMLP2, self).__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.mixed_op = ops.mixedInverseAutoencoder(hidden_size, hidden_c_candidates, drop_p, off_act)

    def forward(self, x, alphas):
        out = self.ln(x)
        out = self.mixed_op(out, alphas)
        return out + x
