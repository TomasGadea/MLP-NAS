import numpy as np
import os
import json
import torch
import random
from torch.distributed import init_process_group

def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
       world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


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
        supernet_params = json.load(open(os.path.join(args.path_to_supernet, 'params.json')))
        from models import SearchController, FixedMixer
        search_model = SearchController(
            device=args.device,
            in_channels=3,
            img_size=args.size,
            hidden_size=supernet_params["hidden_size"],
            patch_size=supernet_params["patch_size"],
            hidden_s_candidates=supernet_params["hidden_s_candidates"],
            hidden_c_candidates=supernet_params["hidden_c_candidates"],
            n_cells=supernet_params["n_cells"],
            num_classes=args.num_classes,
            drop_p=supernet_params["drop_p"],
            off_act=supernet_params["off_act"],
            is_cls_token=supernet_params["is_cls_token"]
        )
        sd = torch.load(os.path.join(args.path_to_supernet, 'W.pt'))
        sd_filtered = {k: v for k,v in sd.items() if 'clf' not in k}
        search_model.load_state_dict(sd_filtered, strict=False)
        alphas = search_model.get_detached_alphas(aslist=False, activated=False)
        model = FixedMixer(search_model.net, alphas)

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
    path = os.path.join(args.output, args.experiment)
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
