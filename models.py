import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import torchsummary
import cell as C


class SearchController(nn.Module):
    def __init__(self,
                 device,
                 in_channels=3,
                 img_size=32,
                 patch_size=4,
                 hidden_size=512,
                 hidden_s_candidates=[256],
                 hidden_c_candidates=[2048],
                 n_cells=8,
                 num_classes=10,
                 drop_p=0.,
                 off_act=False,
                 is_cls_token=False,
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.hidden_s_candidates = hidden_s_candidates
        self.hidden_c_candidates = hidden_c_candidates
        self.n_cells = n_cells
        self.num_classes = num_classes
        self.drop_p = drop_p
        self.off_act = off_act
        self.is_cls_token = is_cls_token

        self.criterion = nn.CrossEntropyLoss().to(device)

        self.alphas = nn.ParameterList()
        for _ in range(n_cells):
            cell_weights = nn.ParameterList()
            cell_weights.append(nn.Parameter(1e-3 * torch.randn(len(hidden_s_candidates)), requires_grad=True))
            cell_weights.append(nn.Parameter(1e-3 * torch.randn(len(hidden_c_candidates)), requires_grad=True))
            # cell_weights.append(nn.Parameter(1e-3 * torch.randn(2), requires_grad=True))  # alphas for skip connections
            self.alphas.append(cell_weights)

        self._alphas = []
        for n, p in self.named_parameters():
            if 'alpha' in n:
                self._alphas.append((n, p))

        self.net = SearchMixer(in_channels, img_size, patch_size, hidden_size, hidden_s_candidates, hidden_c_candidates,
                 n_cells, num_classes, drop_p, off_act, is_cls_token)

    def forward(self, z):
        # b = self.softmax(self.a)
        # weights = [F.sigmoid(alpha) for alpha in self.alphas]
        return self.net(z, self.alphas)

    def loss(self, X, y):
        logits = self.forward(X)
        return self.criterion(logits, y)

    def mmc(self, gated=True):
        """
        we get: torch.nn.parameter.Parameter
        c --> cell index
        i --> index in range 0..len(hidden_s_candidates)
        j --> index in range 0..len(hidden_c_candidates)

        model.cells[c].mlp1.mixed_op.ops[i][0].weight
        model.cells[c].mlp1.mixed_op.ops[i][3].weight

        model.cells[c].mlp2.mixed_op.ops[j][0].weight
        model.cells[c].mlp2.mixed_op.ops[j][3].weight

        """
        all_W = []
        all_alphas = []
        for c in range(self.n_cells):
            for i in range(len(self.hidden_s_candidates)):
                all_W.append(self.net.cells[c].mlp1.mixed_op.ops[i][0].weight)
                all_W.append(self.net.cells[c].mlp1.mixed_op.ops[i][3].weight)
                all_alphas.append(F.sigmoid(self.alphas[c][0][i]))
                all_alphas.append(F.sigmoid(self.alphas[c][0][i]))
                #         ^
                #         |___ append second time since in and out W belong to the same mixed_op with the same alpha
            for j in range(len(self.hidden_c_candidates)):
                all_W.append(self.net.cells[c].mlp2.mixed_op.ops[j][0].weight)
                all_W.append(self.net.cells[c].mlp2.mixed_op.ops[j][3].weight)
                all_alphas.append(F.sigmoid(self.alphas[c][1][j]))
                all_alphas.append(F.sigmoid(self.alphas[c][1][j]))
                #         ^
                #         |___ append second time since in and out W belong to the same mixed_op with the same alpha
        all_W.append(self.net.clf.weight)

        #all_W = [getattr(self, f'U{i}{j}').weight for j in range(1, self.n_nodes - 1) for i in range(0, j)]
        # all_W = [getattr(self, f'U{i}').weight for i in range(self.n_nodes - 1)]

        if gated:
            row_reg = torch.cat([a * torch.norm(W, p=2, dim=1) for a, W in zip(all_alphas, all_W)])  # 1 x (L*H)
        else:
            row_reg = torch.cat([torch.norm(W, p=2, dim=1) for W in all_W])  # 1 x (L*H)

        reg = torch.norm(row_reg, p=1)  # 1 x 1
        return reg

    def friction(self):
        frict = 0.
        for c in range(self.n_cells):
            for i in range(len(self.hidden_s_candidates)):
                frict += F.sigmoid(self.alphas[c][0][i]) * self.net.cells[c].mlp1.mixed_op.ops[i][0].out_features
            for j in range(len(self.hidden_c_candidates)):
                frict += F.sigmoid(self.alphas[c][1][j]) * self.net.cells[c].mlp2.mixed_op.ops[j][0].out_features
        return frict

    def get_alphas(self):
        for n, p in self._alphas:
            yield p

    def get_named_alphas(self):
        for n, p in self._alphas:
            yield n, p

    def weights(self):
        return self.net.parameters()

    def get_detached_alphas(self, aslist=False, th=None, activated=True, binarize=True, top_k=None):
        detached = []
        for a in self.alphas:
            if isinstance(a, torch.Tensor):
                if activated:
                    d = F.sigmoid(a.detach())
                else:
                    d = a.detach()
                if th is not None:
                    d = torch.where(d >= th, 1, 0)
                if aslist:
                    d = d.tolist()
            elif isinstance(a, nn.ParameterList):
                d = []
                for p in a:
                    if isinstance(p, torch.Tensor):
                        if activated:
                            p = F.sigmoid(p.detach())
                        else:
                            p = p.detach()
                        if th is not None:
                            p = torch.where(p > th, p, 0)
                        if top_k is not None:
                            assert(top_k <= len(p))
                            least_k = len(p) - top_k
                            _, idxs = p.topk(least_k, largest=False)
                            p[idxs] = 0.
                        if binarize:
                            if th is not None:
                                p = torch.where(p > th, 1, 0)
                            else:
                                p = torch.where(p > 0, 1, 0)
                        if aslist:
                            d.append(p.tolist())
                        else:
                            d.append(p.detach())
            detached.append(d)
        return detached


class SearchMixer(nn.Module):
    def __init__(self, in_channels, img_size, patch_size, hidden_size, hidden_s_candidates, hidden_c_candidates,
                 n_cells, num_classes, drop_p, off_act, is_cls_token, fixed_alphas=None):
        super(SearchMixer, self).__init__()
        num_patches = img_size // patch_size * img_size // patch_size
        # (b, c, h, w) -> (b, d, h//p, w//p) -> (b, h//p*w//p, d)
        self.is_cls_token = is_cls_token

        self.patch_emb = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size ,kernel_size=patch_size, stride=patch_size),
            Rearrange('b d h w -> b (h w) d')
        )

        if self.is_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
            num_patches += 1

        self.n_cells = n_cells
        self.hidden_s_candidates = hidden_s_candidates
        self.hidden_c_candidates = hidden_c_candidates
        self.cells = nn.ModuleList()
        for i in range(n_cells):
            cell_fixed_alphas = fixed_alphas[i] if fixed_alphas is not None else None
            cell = C.SearchCellMixer(num_patches, hidden_size, hidden_s_candidates, hidden_c_candidates, drop_p, off_act, cell_fixed_alphas)
            self.cells.append(cell)

        self.ln = nn.LayerNorm(hidden_size)

        self.clf = nn.Linear(hidden_size, num_classes)

    def forward(self, x, alphas):
        out = self.patch_emb(x)
        if self.is_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0),1,1), out], dim=1)
        for k, cell in enumerate(self.cells):
            out = cell(out, alphas[k])
        out = self.ln(out)
        out = out[:, 0] if self.is_cls_token else out.mean(dim=1)
        out = self.clf(out)
        return out


class FixedMixer(nn.Module):
    def __init__(self, in_channels, img_size, patch_size, hidden_size, hidden_s_candidates, hidden_c_candidates,
                 n_cells, num_classes, drop_p, off_act, is_cls_token, fixed_alphas):
        super(FixedMixer, self).__init__()
        self.model = SearchMixer(in_channels, img_size, patch_size, hidden_size, hidden_s_candidates,
                                 hidden_c_candidates, n_cells, num_classes, drop_p, off_act, is_cls_token,
                                 fixed_alphas)
        self.alphas = fixed_alphas

    def forward(self, x):
        return self.model(x, self.alphas)

    def mmc(self, gated=True):
        all_W = []
        all_alphas = []
        for c in range(self.model.n_cells):
            for i in range(len(self.model.hidden_s_candidates)):
                try:
                    all_W.append(self.model.cells[c].mlp1.mixed_op.ops[i][0].weight)
                except:
                    pass
                try:
                    all_W.append(self.model.cells[c].mlp1.mixed_op.ops[i][3].weight)
                except:
                    pass
                try:
                    all_alphas.append(F.sigmoid(self.alphas[c][0][i]))
                except:
                    pass
                try:
                    all_alphas.append(F.sigmoid(self.alphas[c][0][i]))
                except:
                    pass
                #         ^
                #         |___ append second time since in and out W belong to the same mixed_op with the same alpha
            for j in range(len(self.model.hidden_c_candidates)):
                try:
                    all_W.append(self.model.cells[c].mlp2.mixed_op.ops[j][0].weight)
                except:
                    pass
                try:
                    all_W.append(self.model.cells[c].mlp2.mixed_op.ops[j][3].weight)
                except:
                    pass
                try:
                    all_alphas.append(F.sigmoid(self.alphas[c][1][j]))
                except:
                    pass
                try:
                    all_alphas.append(F.sigmoid(self.alphas[c][1][j]))
                except:
                    pass
                #         ^
                #         |___ append second time since in and out W belong to the same mixed_op with the same alpha
        all_W.append(self.model.clf.weight)

        #all_W = [getattr(self, f'U{i}{j}').weight for j in range(1, self.n_nodes - 1) for i in range(0, j)]
        # all_W = [getattr(self, f'U{i}').weight for i in range(self.n_nodes - 1)]

        if gated:
            row_reg = torch.cat([a * torch.norm(W, p=2, dim=1) for a, W in zip(all_alphas, all_W)])  # 1 x (L*H)
        else:
            row_reg = torch.cat([torch.norm(W, p=2, dim=1) for W in all_W])  # 1 x (L*H)

        reg = torch.norm(row_reg, p=1)  # 1 x 1
        return reg










class MLPMixer(nn.Module):
    def __init__(self,in_channels=3,img_size=32, patch_size=4, hidden_size=128, hidden_s=64, hidden_c=512, num_layers=8, num_classes=10, drop_p=0., off_act=False, is_cls_token=False):
        super(MLPMixer, self).__init__()
        num_patches = img_size // patch_size * img_size // patch_size
        # (b, c, h, w) -> (b, d, h//p, w//p) -> (b, h//p*w//p, d)
        self.is_cls_token = is_cls_token
        self.num_layers = num_layers

        self.patch_emb = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size ,kernel_size=patch_size, stride=patch_size),
            Rearrange('b d h w -> b (h w) d')
        )

        if self.is_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
            num_patches += 1


        self.mixer_layers = nn.Sequential(
            *[
                MixerLayer(num_patches, hidden_size, hidden_s, hidden_c, drop_p, off_act)
            for _ in range(num_layers)
            ]
        )
        self.ln = nn.LayerNorm(hidden_size)

        self.clf = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.patch_emb(x)
        if self.is_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0),1,1), out], dim=1)
        out = self.mixer_layers(out)
        out = self.ln(out)
        out = out[:, 0] if self.is_cls_token else out.mean(dim=1)
        out = self.clf(out)
        return out

    def mmc(self):
        """
        we get: torch.nn.parameter.Parameter
        c --> cell index
        i --> index in range 0..len(hidden_s_candidates)
        j --> index in range 0..len(hidden_c_candidates)

        model.cells[c].mlp1.mixed_op.ops[i][0].weight
        model.cells[c].mlp1.mixed_op.ops[i][3].weight

        model.cells[c].mlp2.mixed_op.ops[j][0].weight
        model.cells[c].mlp2.mixed_op.ops[j][3].weight

        """
        all_W = []
        for l in range(self.num_layers):
            all_W.append(self.mixer_layers[l].mlp1.fc1.weight)
            all_W.append(self.mixer_layers[l].mlp1.fc2.weight)
            all_W.append(self.mixer_layers[l].mlp2.fc1.weight)
            all_W.append(self.mixer_layers[l].mlp2.fc2.weight)
        all_W.append(self.clf.weight)

        row_reg = torch.cat([torch.norm(W, p=2, dim=1) for W in all_W])  # 1 x (L*H)
        reg = torch.norm(row_reg, p=1)  # 1 x 1
        return reg

    def friction(self):
        F = 0.
        for l in range(self.num_layers):
            F += self.mixer_layers[l].mlp1.fc1.out_features
            F += self.mixer_layers[l].mlp2.fc1.out_features
        return F


class MixerLayer(nn.Module):
    def __init__(self, num_patches, hidden_size, hidden_s, hidden_c, drop_p, off_act):
        super(MixerLayer, self).__init__()
        self.mlp1 = MLP1(num_patches, hidden_s, hidden_size, drop_p, off_act)
        self.mlp2 = MLP2(hidden_size, hidden_c, drop_p, off_act)
    def forward(self, x):
        out = self.mlp1(x)
        out = self.mlp2(out)
        return out

class MLP1(nn.Module):
    def __init__(self, num_patches, hidden_s, hidden_size, drop_p, off_act):
        super(MLP1, self).__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.T = Rearrange('b s c -> b c s')  # Transpose token and channel axis only
        self.fc1 = nn.Linear(num_patches, hidden_s)
        self.do1 = nn.Dropout(p=drop_p)
        self.fc2 = nn.Linear(hidden_s, num_patches)
        self.do2 = nn.Dropout(p=drop_p)
        self.act = F.gelu if not off_act else lambda x: x

    def forward(self, x):
        out = self.do1(self.act(self.fc1(self.T(self.ln(x)))))
        out = self.T(self.do2(self.fc2(out)))
        return out + x

class MLP2(nn.Module):
    def __init__(self, hidden_size, hidden_c, drop_p, off_act):
        super(MLP2, self).__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_c)
        self.do1 = nn.Dropout(p=drop_p)
        self.fc2 = nn.Linear(hidden_c, hidden_size)
        self.do2 = nn.Dropout(p=drop_p)
        self.act = F.gelu if not off_act else lambda x:x
    def forward(self, x):
        out = self.do1(self.act(self.fc1(self.ln(x))))
        out = self.do2(self.fc2(out))
        return out+x
