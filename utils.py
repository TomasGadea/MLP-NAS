import numpy as np
import os
import json
import torch

def get_model(args):
    model = None
    if args.model == 'search_mixer':
        from models import SearchController
        model = SearchController(
            device=args.device,
            in_channels=3,
            img_size=args.size,
            hidden_size=args.hidden_size,
            patch_size=args.patch_size,
            hidden_s_candidates=args.hidden_s_candidates,
            hidden_c_candidates=args.hidden_c_candidates,
            n_cells=args.n_cells,
            num_classes=args.num_classes,
            drop_p=args.drop_p,
            off_act=args.off_act,
            is_cls_token=args.is_cls_token
        )

    elif args.model == 'fixed-mixer':
        from models import SearchController, FixedMixer
        search_model = SearchController(
            device=args.device,
            in_channels=3,
            img_size=args.size,
            hidden_size=args.hidden_size,
            patch_size=args.patch_size,
            hidden_s_candidates=args.hidden_s_candidates,
            hidden_c_candidates=args.hidden_c_candidates,
            n_cells=args.n_cells,
            num_classes=args.num_classes,
            drop_p=args.drop_p,
            off_act=args.off_act,
            is_cls_token=args.is_cls_token
        )
        search_model.load_state_dict(
            torch.load(os.path.join(args.path, f"W-search_mixer_{args.dataset}_last.pt"))
        )
        alphas = search_model.get_detached_alphas(aslist=True, activated=False)
        model = search_model.net
        return model.to(args.device), alphas

    elif args.model == 'mlp-mixer':
        from models import MLPMixer
        model = MLPMixer(
            in_channels=3,
            img_size=args.size,
            patch_size=args.patch_size,
            hidden_size=args.hidden_size,
            hidden_s=args.hidden_s,
            hidden_c=args.hidden_c,
            num_layers=args.num_layers,
            num_classes=args.num_classes,
            drop_p=args.drop_p,
            off_act=args.off_act,
            is_cls_token=args.is_cls_token
        )
    else:
        raise ValueError(f"No such model: {args.model}")

    return model.to(args.device)


def save_config(args):
    config = args.__dict__.copy()
    config['device'] = config['device'].__str__()
    path = os.path.join(args.output, args.project, args.experiment)
    if args.model == 'fixed-mixer':
        path = os.path.join(path, f"{args.model}_{args.subexperiment}")
    os.makedirs(path, exist_ok=True)
    with open(path + '/params.json', 'w') as ff:
        json.dump(config, ff, indent=4, sort_keys=True)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int64(W * cut_rat)
    cut_h = np.int64(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
