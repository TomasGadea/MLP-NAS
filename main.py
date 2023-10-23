import argparse
from datetime import datetime
import torch
import random
import wandb
import os
import json
config = json.load(open('api_key.config'))
os.environ["WANDB_API_KEY"] = config['WANDB_API_KEY']
wandb.login(key=os.environ['WANDB_API_KEY'])

from dataloader import get_dataloaders
from utils import get_model, save_config
from train import Trainer
from fixed_main import main as f_main

def main(args):
    print(f"PID: {os.getpid()}")
    with wandb.init(project=args.project, config=args, name=args.experiment):
        save_config(args)
        train_dl, valid_dl, test_dl = get_dataloaders(args)
        model = get_model(args)
        trainer = Trainer(model, args)
        trainer.fit(train_dl, valid_dl, test_dl, args)

    if args.retrain_fixed:
        args.path = os.path.join(args.output, args.project, args.experiment)
        f_main(args)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--project', type=str, default=f"search_mixer")
    parser.add_argument('--output', type=str, default=f"./out")
    parser.add_argument('--experiment', type=str, default=f"experiment_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
    parser.add_argument('--seed', type=int, default=random.randint(1, 1e5))
    parser.add_argument('--dataset', required=True, choices=['c10', 'c100', 'svhn', 'stl10'])
    parser.add_argument('--valid-ratio', type=float, default=0.5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--eval-batch-size', type=int, default=1024)
    parser.add_argument('--autoaugment', action='store_true')

    # model
    parser.add_argument('--model', required=True, choices=['mlp_mixer', 'search_mixer'])
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--patch-size', type=int, default=4)
    parser.add_argument('--hidden-s-candidates', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument('--hidden-c-candidates', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument('--n-cells', type=int, default=8)
    parser.add_argument('--drop-p', type=float, default=0.)
    parser.add_argument('--off-act', action='store_true', help='Disable activation function')
    parser.add_argument('--is-cls-token', action='store_true', help='Introduce a class token.')

    # training
    parser.add_argument('--clip-grad', type=float, default=0, help="0 means disabling clip-grad")
    parser.add_argument('--cutmix-beta', type=float, default=1.0)
    parser.add_argument('--cutmix-prob', type=float, default=0.)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--w01', type=float, default=1.)
    parser.add_argument('--mu', type=float, default=0.)
    parser.add_argument('--l', type=float, default=0.)

    parser.add_argument('--w-optimizer', default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--w-lr', type=float, default=1e-3)
    parser.add_argument('--w-momentum', type=float, default=0.9)
    parser.add_argument('--w-weight-decay', type=float, default=5e-5)
    parser.add_argument('--off-nesterov', action='store_true')
    parser.add_argument('--w-beta1', type=float, default=0.9)
    parser.add_argument('--w-beta2', type=float, default=0.99)

    parser.add_argument('--a-optimizer', default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--a-lr', type=float, default=1e-3)
    parser.add_argument('--a-beta1', type=float, default=0.9)
    parser.add_argument('--a-beta2', type=float, default=0.99)
    parser.add_argument('--a-weight-decay', type=float, default=5e-5)


    parser.add_argument('--w-scheduler', default='cosine', choices=['step', 'cosine'])
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--w-gamma', type=float, default=0.1)
    parser.add_argument('--w-min-lr', type=float, default=1e-6)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--unrolled', action='store_true')





    # Retrain Fixed Arch
    parser.add_argument('--retrain-fixed', action='store_true')
    parser.add_argument('--fixed-batch-size', type=int, default=128)
    parser.add_argument('--fixed-eval-batch-size', type=int, default=1024)
    parser.add_argument('--fixed-seed', type=int, default=random.randint(1, 1e5))
    parser.add_argument('--fixed-epochs', type=int, default=300)
    parser.add_argument('--fixed-th', type=float, default=0.5)

    parser.add_argument('--fixed-lr', type=float, default=1e-3)
    parser.add_argument('--fixed-min-lr', type=float, default=1e-6)
    parser.add_argument('--fixed-momentum', type=float, default=0.9)
    parser.add_argument('--fixed-optimizer', default='adam', choices=['Adam', 'SGD'])
    parser.add_argument('--fixed-scheduler', default='cosine', choices=['step', 'cosine'])
    parser.add_argument('--fixed-beta1', type=float, default=0.9)
    parser.add_argument('--fixed-beta2', type=float, default=0.99)
    parser.add_argument('--fixed-weight-decay', type=float, default=5e-5)
    parser.add_argument('--fixed-off-nesterov', action='store_true')
    parser.add_argument('--fixed-label-smoothing', type=float, default=0.1)
    parser.add_argument('--fixed-gamma', type=float, default=0.1)
    parser.add_argument('--fixed-warmup-epoch', type=int, default=5)
    parser.add_argument('--fixed-autoaugment', action='store_true')
    parser.add_argument('--fixed-clip-grad', type=float, default=0, help="0 means disabling clip-grad")
    parser.add_argument('--fixed-cutmix-beta', type=float, default=1.0)
    parser.add_argument('--fixed-cutmix-prob', type=float, default=0.)
    parser.add_argument('--discrete', action='store_true')


    args = parser.parse_args()
    args.hidden_s_candidates = list(map(lambda x: int(x), args.hidden_s_candidates))
    args.hidden_c_candidates = list(map(lambda x: int(x), args.hidden_c_candidates))

    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.nesterov = not args.off_nesterov
    args.pid = os.getpid()
    args.subexperiment=f"experiment_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
    torch.random.manual_seed(args.seed)

    main(args)

