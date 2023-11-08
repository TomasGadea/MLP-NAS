import torch.nn as nn
import torch
import torch.nn.functional as F


class mixedInverseAutoencoder(nn.Module):
    def __init__(self, in_dim: int, search_dims: list[int], drop_p: float, off_act: bool, fixed_alphas):
        super().__init__()
        alphas = fixed_alphas if fixed_alphas is not None else torch.ones(len(search_dims))
        self.are_activated = fixed_alphas is not None

        self.ops = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, k),
                nn.GELU() if not off_act else lambda x: x,
                nn.Dropout(p=drop_p),
                nn.Linear(k, in_dim),
                nn.Dropout(p=drop_p)
            ) for i, k in enumerate(search_dims) if alphas[i] > 0.
        ])

    def forward(self, x, alphas):
        weights = F.sigmoid(alphas) if not self.are_activated else alphas
        f = sum(w * op(x) for w, op in zip(weights, self.ops))
        if isinstance(f, int): return 0. * x
        return f


class mask_op(nn.Module):
    def __init__(self, n, max_dim, dim):
        super().__init__()
        self.n = n
        self.max_dim = max
        self.dim = dim
        self.mask = torch.cat((
            torch.ones((n, dim)),
            torch.zeros((n, max_dim - dim))
        ), dim=-1)

        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def forward(self, x):
        return torch.mul(x, self.mask)


class ProjMixedOp(nn.Module):
    """ Mixed operation """

    def __init__(self, in_dim, search_out_dims, seq_len, bias):
        """
        :param in_dim: fixed input dim of linear layer.
        :param seq_len: number of rows in input matrix.
        :param bias: boolean whether to add bias or not.
        :param search_out_dims: list of searched output dimensions.
        """
        super().__init__()
        self.search_out_dims = search_out_dims
        self.lin = nn.Linear(in_dim, max(search_out_dims), bias=bias)
        # self.norm = nn.BatchNorm1d(num_features=max_hidden_size)  # 1 channel only for grayscaled inputs
        self.ops = nn.ModuleList([
            nn.Sequential(
                # nn.Linear(max_hidden_size, max_hidden_size, bias=bias),
                self.lin,
                # nn.ReLU(),
                mask_op(seq_len, max(search_out_dims), search_out_dims[k])
            ) for k in range(len(search_out_dims))
        ])

    def forward(self, x, alphas):
        """
        Args:
            x: input
            weights: weight for each operation
        """
        weights = F.sigmoid(alphas)
        f = sum(w * op(x) for w, op in zip(weights, self.ops))
        return f


def masked_chunk(x, n, max_dim, dim):
    x_trunc = x[:, :, :dim]
    res_trunc, gate_trunc = x_trunc.chunk(2, dim=-1)

    res = torch.cat((
        res_trunc,
        torch.zeros((n, max_dim // 2 - dim // 2)).cuda()
    ), dim=-1)

    gate = torch.cat((
        gate_trunc,
        torch.zeros((n, max_dim // 2 - dim // 2)).cuda()
    ), dim=-1)

    return res, gate
