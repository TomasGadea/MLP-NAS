import sys
sys.path.append('./AutoAugment/')

import torch
import torchvision
import torchvision.transforms as transforms
from AutoAugment.autoaugment import CIFAR10Policy, SVHNPolicy, ImageNetPolicy
import numpy as np
import json
import os
from torch.utils.data.distributed import DistributedSampler
from timm.data.transforms_factory import transforms_imagenet_train, transforms_imagenet_eval


def get_dataloaders(args):
    g = torch.Generator()
    g.manual_seed(args.seed)

    train_transform, test_transform = get_transform(args)

    if args.dataset == "c10":
        train_ds = torchvision.datasets.CIFAR10('./datasets', train=True, transform=train_transform, download=True)
        test_ds = torchvision.datasets.CIFAR10('./datasets', train=False, transform=test_transform, download=True)
        args.num_classes = 10
    elif args.dataset == "c100":
        train_ds = torchvision.datasets.CIFAR100('./datasets', train=True, transform=train_transform, download=True)
        test_ds = torchvision.datasets.CIFAR100('./datasets', train=False, transform=test_transform, download=True)
        args.num_classes = 100
    elif args.dataset == "svhn":
        train_ds = torchvision.datasets.SVHN('./datasets', split='train', transform=train_transform, download=True)
        test_ds = torchvision.datasets.SVHN('./datasets', split='test', transform=test_transform, download=True)
        args.num_classes = 10
    elif args.dataset == 'stl10':
        train_ds = torchvision.datasets.STL10('./datasets', split='train', transform=train_transform, download=True)
        test_ds = torchvision.datasets.STL10('./datasets', split='test', transform=test_transform, download=True)
        args.num_classes = 10

    elif args.dataset == 'imagenet':
        config = json.load(open('config.json'))
        traindir = os.path.join(config['IMAGENET_PATH'], 'train')
        valdir = os.path.join(config['IMAGENET_PATH'], 'val')

        train_ds = torchvision.datasets.ImageFolder(traindir, transform=train_transform)
        test_ds = torchvision.datasets.ImageFolder(valdir, transform=test_transform)
        args.num_classes = 1000
    elif args.dataset == 'imagenet100':
        config = json.load(open('config.json'))
        traindir = os.path.join(config['IMAGENET100_PATH'], 'train')
        valdir = os.path.join(config['IMAGENET100_PATH'], 'val')

        train_ds = torchvision.datasets.ImageFolder(traindir, transform=train_transform)
        test_ds = torchvision.datasets.ImageFolder(valdir, transform=test_transform)
        args.num_classes = 100

    else:
        raise ValueError(f"No such dataset:{args.dataset}")

    print(f"args.valid_ratio: {args.valid_ratio}")
    if args.valid_ratio > 0:
        num_train = len(train_ds)
        indices = list(range(num_train))
        split = int(np.floor(args.valid_ratio * num_train))

        train_dl = torch.utils.data.DataLoader(
            train_ds, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split], generator=g),
            num_workers=args.num_workers, pin_memory=True, generator=g)

        valid_dl = torch.utils.data.DataLoader(
            train_ds, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train], generator=g),
            num_workers=args.num_workers, pin_memory=True, generator=g)

    else:
        if args.distributed:
            train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                                                   sampler=DistributedSampler(train_ds), num_workers=args.num_workers,
                                                   pin_memory=True, generator=g)
        else:
            train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.num_workers, pin_memory=True, generator=g)
        valid_dl = None

    if args.distributed:
        test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False,
                                              sampler=DistributedSampler(test_ds), num_workers=args.num_workers,
                                              pin_memory=True, generator=g)
    else:
        test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True, generator=g)


    return train_dl, valid_dl, test_dl


def get_transform(args):
    if args.dataset in ["c10", "c100", 'svhn', 'stl10']:
        if not hasattr(args, 'padding') or args.padding is None:
            args.padding = 4
        if not hasattr(args, 'size') or args.size is None:
            args.size = 32
        if args.dataset == "c10":
            args.mean, args.std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
        elif args.dataset == "c100":
            args.mean, args.std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        elif args.dataset == "svhn":
            args.mean, args.std = [0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970]
        elif args.dataset == 'stl10':
            args.mean, args.std = [0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616]
            args.size = 96

    elif args.dataset == 'imagenet':
        args.padding = 28
        args.size = 224
        args.mean, args.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    elif args.dataset == 'imagenet100':
        args.padding = 28
        args.size = 224
        args.mean, args.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        if args.use_timm_transform:
            train_transform = transforms_imagenet_train(
                img_size=224,
                scale=None,
                ratio=None,
                hflip=args.hflip,
                vflip=args.vflip,
                color_jitter=args.color_jitter,
                auto_augment=args.autoaugment,
                interpolation=args.train_interpolation,
                use_prefetcher=False,
                mean=args.mean,
                std=args.std,
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
                re_num_splits=0,
                separate=False,
                force_color_jitter=False
            )
            test_transform = transforms_imagenet_eval(
                img_size=224,
                crop_pct=None,
                crop_mode=None,
                interpolation=args.train_interpolation,
                use_prefetcher=False,
                mean=args.mean,
                std=args.std
            )
            return train_transform, test_transform


    train_transform_list = [transforms.Resize(size=(args.size, args.size))]
    if args.dataset not in ["svhn", "imagenet"]:
        train_transform_list.append(transforms.RandomCrop(size=(args.size, args.size), padding=args.padding))

    if args.autoaugment:
        if args.dataset == 'c10' or args.dataset == 'c100':
            train_transform_list.append(CIFAR10Policy())
        elif args.dataset == 'svhn':
            train_transform_list.append(SVHNPolicy())
        elif args.dataset == 'imagenet':
            train_transform_list.append(ImageNetPolicy())
        else:
            print(f"No AutoAugment for {args.dataset}")   

    train_transform = transforms.Compose(
        train_transform_list+[
            transforms.ToTensor(),
            transforms.Normalize(
                mean=args.mean,
                std=args.std
            )
        ]
    )

    if  args.dataset == 'imagenet' or args.dataset =='imagenet100':
        test_transform_list = [transforms.Resize(size=(args.size, args.size))]
        test_transform = transforms.Compose(
                test_transform_list+[
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=args.mean,
                        std=args.std
                    )
                ]
            )

    else:
        test_transform_list = [transforms.Resize(size=(args.size, args.size))]
        test_transform = transforms.Compose(
                test_transform_list+[
            transforms.ToTensor(),
            transforms.Normalize(
                mean=args.mean,
                std=args.std
            )
        ])

    return train_transform, test_transform
