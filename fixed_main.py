import argparse
from datetime import datetime
import torch
import random
import wandb
import os
import json

from dataloader import get_dataloaders
from utils import get_model, save_config
from train import Trainer, VanillaTrainer


def main(args):
    print(f"PID: {os.getpid()}")
    name = f"{args.model}_{args.experiment}"
    if args.model == 'fixed-mixer':
        name += f"_{args.subexperiment}"

    if args.wandb:
        config = json.load(open('api_key.config'))
        os.environ["WANDB_API_KEY"] = config['WANDB_API_KEY']
        wandb.login(key=os.environ['WANDB_API_KEY'])
        wandb.init(project=args.project, config=args, name=name)

    save_config(args)
    train_dl, valid_dl, test_dl = get_dataloaders(args)
    fixed_model = get_model(args)
    v_trainer = VanillaTrainer(fixed_model, args)
    v_trainer.fit(train_dl, valid_dl, test_dl, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--project', type=str, default=f"new_project")
    parser.add_argument('--path', type=str)
    parser.add_argument('--model', type=str, default=f"fixed-mixer", choices=['fixed-mixer', 'mlp-mixer'])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--eval-batch-size', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=random.randint(1, 1e5))
    parser.add_argument('--dataset', required=True, choices=['c10', 'c100', 'svhn', 'stl10'])
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--autoaugment', action='store_true')
    parser.add_argument('--valid-ratio', type=float, default=0.5)

    # Retrain Discrete Arch
    parser.add_argument('--fixed-epochs', type=int, default=300)
    parser.add_argument('--fixed-lr', type=float, default=1e-3)
    parser.add_argument('--fixed-min-lr', type=float, default=1e-6)
    parser.add_argument('--fixed-momentum', type=float, default=0.9)
    parser.add_argument('--fixed-optimizer', default='SGD', choices=['Adam', 'SGD'])
    parser.add_argument('--fixed-scheduler', default='cosine', choices=['step', 'cosine'])
    parser.add_argument('--fixed-beta1', type=float, default=0.9)
    parser.add_argument('--fixed-beta2', type=float, default=0.99)
    parser.add_argument('--fixed-weight-decay', type=float, default=5e-5)
    parser.add_argument('--fixed-off-nesterov', action='store_true')
    parser.add_argument('--fixed-label-smoothing', type=float, default=0.1)
    parser.add_argument('--fixed-gamma', type=float, default=0.1)
    parser.add_argument('--fixed-warmup-epoch', type=int, default=5)
    parser.add_argument('--fixed-clip-grad', type=float, default=0, help="0 means disabling clip-grad")
    parser.add_argument('--fixed-cutmix-beta', type=float, default=1.0)
    parser.add_argument('--fixed-cutmix-prob', type=float, default=0.)

    parser.add_argument('--patch-size', type=int, default=4)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--hidden-s', type=int, default=64)
    parser.add_argument('--hidden-c', type=int, default=512)
    parser.add_argument('--num-layers', type=int, default=8)
    parser.add_argument('--drop-p', type=float, default=0.)
    parser.add_argument('--off-act', action='store_true')
    parser.add_argument('--is-cls-token', action='store_true')

    parser.add_argument('--discrete', action='store_true')
    parser.add_argument('--discrete-th', type=float, default=0.5)

    parser.add_argument('--wandb', action='store_true')


    args = parser.parse_args()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.pid = os.getpid()
    args.subexperiment=f"experiment_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
    print("ARGS: ", args)

    torch.random.manual_seed(args.seed)

    if args.model == 'mlp-mixer':
        args.experiment = f"experiment_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
        args.output = "./out"
    else:
        params = json.load(open(os.path.join(args.path, 'params.json')))
        for k, v in params.items():
            if k not in args:
                setattr(args, k, v)

    main(args)


