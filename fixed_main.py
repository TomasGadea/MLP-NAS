import argparse
from datetime import datetime
import torch
import random
import wandb
import os
import json

from dataloader import get_dataloaders
from utils import get_model, save_config, ddp_setup
from torch.distributed import destroy_process_group
import torch.multiprocessing as mp
from train import Trainer, VanillaTrainer
from fvcore.nn import FlopCountAnalysis, parameter_count, flop_count_table


def main(args):
    print(f"PID: {os.getpid()}")

    if args.wandb:
        name = f"{args.model}_{args.experiment}"
        if args.model == 'fixed-mixer':
            name += f"_{args.subexperiment}"
        config = json.load(open('config.json'))
        os.environ["WANDB_API_KEY"] = config['WANDB_API_KEY']
        wandb.login(key=os.environ['WANDB_API_KEY'])
        wandb.init(project=args.project, config=args, name=name)

    train_dl, valid_dl, test_dl = get_dataloaders(args)
    fixed_model = get_model(args)

    # flops
    input = torch.rand((1, 3, args.size, args.size)).to(args.device)
    flops = FlopCountAnalysis(fixed_model, input)
    args.flops = flops.total()
    args.n_params = parameter_count(fixed_model)['']
    os.makedirs(os.path.join(args.output, args.experiment), exist_ok=True)
    with open(os.path.join(args.output, args.experiment, "flops_table.txt"), "w") as text_file:
        text_file.write(f"{flop_count_table(flops, max_depth=0)}")

    save_config(args)
    v_trainer = VanillaTrainer(fixed_model, args)
    v_trainer.fit(train_dl, valid_dl, test_dl, args)


def distributed_main(rank, world_size, args):
    print(f"PID: {os.getpid()}")
    args.device = rank
    args.world_size = world_size
    ddp_setup(rank, world_size)

    if args.wandb:
        name = f"{args.model}_{args.experiment}"
        if args.model == 'fixed-mixer':
            name += f"_{args.subexperiment}"
        config = json.load(open('config.json'))
        os.environ["WANDB_API_KEY"] = config['WANDB_API_KEY']
        wandb.login(key=os.environ['WANDB_API_KEY'])
        wandb.init(project=args.project, config=args, name=name)

    train_dl, valid_dl, test_dl = get_dataloaders(args)
    fixed_model = get_model(args)

    # flops
    input = torch.rand((1, 3, args.size, args.size)).to(args.device)
    flops = FlopCountAnalysis(fixed_model, input)
    args.flops = flops.total()
    args.n_params = parameter_count(fixed_model)['']
    os.makedirs(os.path.join(args.output, args.experiment), exist_ok=True)
    with open(os.path.join(args.output, args.experiment, "flops_table.txt"), "w") as text_file:
        text_file.write(f"{flop_count_table(flops, max_depth=0)}")

    save_config(args)
    v_trainer = VanillaTrainer(fixed_model, args)
    v_trainer.fit(train_dl, valid_dl, test_dl, args)

    destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--project', type=str, default=f"new_project")
    parser.add_argument('--output', type=str, default=f"./out/retrain")
    parser.add_argument('--experiment', type=str, default=f"experiment_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
    parser.add_argument('--path-to-supernet', type=str)
    parser.add_argument('--model', type=str, default=f"fixed-mixer", choices=['fixed-mixer', 'mlp-mixer'])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--eval-batch-size', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=random.randint(1, 1e5))
    parser.add_argument('--dataset', required=True, choices=['c10', 'c100', 'svhn', 'stl10', 'imagenet','imagenet100'])
    parser.add_argument('--img-size', type=int, dest='size')
    parser.add_argument('--padding', type=int)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--valid-ratio', type=float, default=0.)
    parser.add_argument('--recovery-interval', type=int, default=0)
    # data aug
    parser.add_argument('--use-timm-transform', action='store_true', help='Use timm.data transforms (for Imagenet only)')
    parser.add_argument('--hflip', type=float, default=0.5, help='Horizontal flip training aug probability')
    parser.add_argument('--vflip', type=float, default=0., help='Vertical flip training aug probability')
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                       help='Color jitter factor (default: 0.4)')
    parser.add_argument('--train-interpolation', type=str, default='bilinear', choices=["bilinear", "bicubic"],
                       help='Training interpolation (random, bilinear, bicubic default: "random")')
    parser.add_argument('--reprob', type=float, default=0., help='Random erase prob (default: 0.)')
    parser.add_argument('--remode', type=str, default='pixel', choices=['pixel', 'rand', 'const'],
                       help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                       help='Random erase count (default: 1)')
    #parser.add_argument('--resplit', action='store_true', default=False,
    #                   help='Do not random erase first (clean) augmentation split')
    parser.add_argument('--autoaugment', type=str, default=None,
                                    choices=["rand", "augmix", "original", "originalr", "v0", "v0r", "3a"])


    parser.add_argument('--th-arch', type=float, default=None, help='threshold to filter out operations, ops with alpha <= th will not be chosen')
    parser.add_argument('--top-k', type=int, default=None, help='the top k operations to select based on their weights. Compatible with --th-arch and --binarize')
    parser.add_argument('--binarize-arch', action='store_true', help='keep original alphas when thresholding or binarize to 0 or 1 with --th-arch value')

    # Retrain Discrete Arch
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--optimizer', default='SGD', choices=['Adam', 'SGD'])
    parser.add_argument('--scheduler', default='cosine', choices=['step', 'cosine'])
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--weight-decay', type=float, default=5e-5)
    parser.add_argument('--off-nesterov', action='store_true')
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--warmup-epoch', type=int, default=5)
    parser.add_argument('--clip-grad', type=float, default=0, help="0 means disabling clip-grad")
    parser.add_argument('--cutmix-beta', type=float, default=1.0)
    parser.add_argument('--cutmix-prob', type=float, default=0.)

    parser.add_argument('--discrete', action='store_true')
    parser.add_argument('--discrete-th', type=float, default=0.5)

    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--verbose', action='store_true')


    args = parser.parse_args()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.nesterov = not args.off_nesterov
    args.pid = os.getpid()

    torch.random.manual_seed(args.seed)


    if args.distributed:
        world_size = torch.cuda.device_count()
        mp.spawn(distributed_main, args=(world_size, args), nprocs=world_size)
    else:
        main(args)


