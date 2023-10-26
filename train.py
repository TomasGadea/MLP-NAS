import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import os
import pandas as pd
import warmup_scheduler
import numpy as np
from tqdm import tqdm, trange
from timm.data.mixup import Mixup
from architect import Architect

from utils import rand_bbox


class Trainer(object):
    def __init__(self, model, args):
        self.model = model

        self.device = args.device
        self.clip_grad = args.clip_grad
        self.cutmix_beta = args.cutmix_beta
        self.cutmix_prob = args.cutmix_prob
        self.label_smoothing = args.label_smoothing
        self.w01 = args.w01
        self.mu = args.mu
        self.l = args.l
        self.unrolled = args.unrolled
        self.n_cells = args.n_cells
        self.hidden_s_candidates = args.hidden_s_candidates
        self.hidden_c_candidates = args.hidden_c_candidates
        self.wandb = args.wandb
        if self.wandb:
            wandb.config.update(args, allow_val_change=True)

        self.mixup_fn = Mixup(
            cutmix_alpha=self.cutmix_beta,
            prob=self.cutmix_prob,
            label_smoothing=self.label_smoothing,
            num_classes=args.num_classes,
        )

        # weights optimizer
        if args.w_optimizer == 'SGD':
            self.w_optimizer = optim.SGD(self.model.parameters(), lr=args.w_lr, momentum=args.w_momentum, weight_decay=args.w_weight_decay, nesterov=args.nesterov)
        elif args.w_optimizer == 'Adam':
            self.w_optimizer = optim.Adam(self.model.parameters(), lr=args.w_lr, betas=(args.w_beta1, args.w_beta2), weight_decay=args.w_weight_decay)
        else:
            raise ValueError(f"No such W optimizer: {args.w_optimizer}")

        # alphas optimizer
        if args.a_optimizer == 'Adam':
            self.a_optimizer = optim.Adam(self.model.parameters(), lr=args.a_lr, betas=(args.a_beta1, args.a_beta2), weight_decay=args.a_weight_decay)
        else:
            raise ValueError(f"No such optimizer: {args.a_optimizer}")

        # weights scheduler
        if args.w_scheduler == 'step':
            self.base_scheduler = optim.lr_scheduler.MultiStepLR(self.w_optimizer, milestones=[args.epochs//2, 3*args.epochs//4], gamma=args.w_gamma)
        elif args.w_scheduler == 'cosine':
            self.base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.w_optimizer, T_max=args.epochs, eta_min=args.w_min_lr)
        else:
            raise ValueError(f"No such scheduler: {args.w_scheduler}")

        if args.warmup_epochs:
            self.scheduler = warmup_scheduler.GradualWarmupScheduler(self.w_optimizer, multiplier=1., total_epoch=args.warmup_epochs, after_scheduler=self.base_scheduler)
        else:
            self.scheduler = self.base_scheduler

        # scaler
        self.scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

        self.epochs = args.epochs

        # Architect
        self.architect = Architect(self.model, args.w_momentum, args.w_weight_decay, self.a_optimizer, use_amp=args.use_amp)

        self.num_steps = 0
        self.epoch_loss, self.epoch_corr, self.epoch_acc = 0., 0., 0.

    def _train_one_step(self, trn_X, trn_y, val_X, val_y, mixup_fn):
        self.model.train()
        self.num_steps += 1

        val_X = val_X.type(torch.FloatTensor)
        val_X, val_y = val_X.to(self.device), val_y.to(self.device)

        trn_X = trn_X.type(torch.FloatTensor)
        trn_X, trn_y = trn_X.to(self.device), trn_y.to(self.device)

        trn_X, trn_y = mixup_fn(trn_X, trn_y)

        w_lr = self.scheduler.get_last_lr()[0]

        # phase 2. architect step (alpha)
        if self.unrolled:
            self.a_optimizer.zero_grad()
            self.architect.unrolled_backward(trn_X, trn_y, val_X, val_y, w_lr, self.w_optimizer, self.w01)
            self.a_optimizer.step()
        else:
            arch_loss, L01_loss, net_loss, friction = self.architect.rolled_backward(val_X, val_y, self.w01, self.l)

        # phase 1. child network step (w)
        self.w_optimizer.zero_grad()
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                logits = self.model(trn_X)
                loss = self.model.criterion(logits, trn_y) + self.mu * self.model.mmc()
            self.scaler.scale(loss).backward()
        else:
            logits = self.model(trn_X)
            loss = self.model.criterion(logits, trn_y) + self.mu * self.model.mmc()
            loss.backward()

        # gradient clipping
        if self.clip_grad:
            nn.utils.clip_grad_norm_(self.model.weights(), self.clip_grad)

        if self.scaler is not None:
            self.scaler.step(self.w_optimizer)
            self.scaler.update()
        else:
            self.w_optimizer.step()



        # update metrics
        acc = logits.argmax(dim=-1).eq(trn_y.argmax(dim=-1)).sum(-1)/len(trn_X)
        """
        wandb.log({
            'loss': loss,
            'acc': acc
        }, step=self.num_steps)
        """

        self.epoch_tr_loss += loss * len(trn_X)
        self.epoch_L01_loss += L01_loss * len(trn_X)
        self.epoch_friction += friction * len(trn_X)
        self.epoch_tr_corr += logits.argmax(dim=-1).eq(trn_y.argmax(dim=-1)).sum(-1)

    # @torch.no_grad
    def _test_one_step(self, val_X, val_y, testing=False):
        self.model.eval()
        val_X, val_y = val_X.to(self.device), val_y.to(self.device)

        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    logits = self.model(val_X)
                    loss = self.model.criterion(logits, val_y)
        else:
            with torch.no_grad():
                logits = self.model(val_X)
                loss = self.model.criterion(logits, val_y)


        if testing:
            self.test_loss += loss * len(val_X)
            self.test_corr += logits.argmax(dim=-1).eq(val_y).sum(-1)
        else:
            self.epoch_loss += loss * len(val_X)
            self.epoch_corr += logits.argmax(dim=-1).eq(val_y).sum(-1)

    def fit(self, train_dl, valid_dl, test_dl, args):
        df = {'epoch': [], 'train_loss': [], 'L01_loss': [], 'valid_loss': [], 'train_acc': [], 'valid_acc': [], 'friction': [], 'mmc': [],  'alphas': [], 'test_loss': [], 'test_acc': []}
        best_acc = 0.
        for epoch in trange(1, self.epochs+1):
            num_tr_imgs = 0.
            self.epoch_tr_loss, self.epoch_L01_loss, self.epoch_friction, self.epoch_tr_corr, self.epoch_tr_acc = 0., 0., 0., 0., 0.

            for batch_idx, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(train_dl, valid_dl)):
                self._train_one_step(trn_X, trn_y, val_X, val_y, self.mixup_fn)
                num_tr_imgs += len(trn_X)

            self.scheduler.step()

            ################################################  LOGGING  #################################################
            self.epoch_tr_loss /= num_tr_imgs
            self.epoch_L01_loss /= num_tr_imgs
            self.epoch_friction /= num_tr_imgs
            self.epoch_tr_acc = self.epoch_tr_corr / num_tr_imgs

            if self.wandb:
                wandb.log({
                    'w_lr': self.scheduler.get_last_lr()[0],
                    'epoch_tr_loss': self.epoch_tr_loss,
                    'epoch_L01_loss': self.epoch_L01_loss,
                    'friction': self.model.friction().item(),
                    'mmc': self.model.mmc().item(),
                    'epoch_tr_acc': self.epoch_tr_acc
                    }, step=epoch
                )
                alphas = self.model.get_detached_alphas(aslist=True)
                for i in range(self.n_cells):
                    for j in range(len(self.hidden_s_candidates)):
                        wandb.log({
                            f'a_cell{i}_Ds_{self.hidden_s_candidates[j]}': alphas[i][0][j],
                        }, step=epoch)
                    for j in range(len(self.hidden_c_candidates)):
                        wandb.log({
                            f'a_cell{i}_Dc_{self.hidden_c_candidates[j]}': alphas[i][1][j],
                        }, step=epoch)

            df['epoch'].append(epoch)
            df['train_acc'].append(self.epoch_tr_acc.item())
            df['train_loss'].append(self.epoch_tr_loss.item())
            df['L01_loss'].append(self.epoch_L01_loss)
            df['friction'].append(self.model.friction().item())
            df['mmc'].append(self.model.mmc().item())
            df['alphas'].append(None)
            ############################################################################################################

            num_imgs = 0.
            self.epoch_loss, self.epoch_corr, self.epoch_acc = 0., 0., 0.

            for batch_idx, (val_X, val_y) in enumerate(valid_dl):
                self._test_one_step(val_X, val_y)
                num_imgs += len(val_X)

            ################################################  LOGGING  #################################################
            self.epoch_loss /= num_imgs
            self.epoch_acc = self.epoch_corr / num_imgs
            if self.wandb:
                wandb.log({
                    'val_loss': self.epoch_loss,
                    'val_acc': self.epoch_acc
                    }, step=epoch
                )
            df['valid_loss'].append(self.epoch_loss.item())
            df['valid_acc'].append(self.epoch_acc.item())
            ############################################################################################################

            if self.epoch_acc > best_acc:
                num_imgs = 0.
                self.test_loss, self.test_corr, self.test_acc = 0., 0., 0.
                for batch_idx, (test_X, test_y) in enumerate(test_dl):
                    self._test_one_step(test_X, test_y, testing=True)
                    num_imgs += len(test_X)
                self.test_loss /= num_imgs
                self.test_acc = self.test_corr / num_imgs
                if self.wandb:
                    wandb.log({
                        'test_loss': self.test_loss,
                        'test_acc': self.test_acc
                    }, step=epoch
                    )
                df['test_loss'].append(self.test_loss.item())
                df['test_acc'].append(self.test_acc.item())
                # save model weights
                path = os.path.join(args.output, args.experiment)
                os.makedirs(path, exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(path, 'W_test.pt'))
            else:
                df['test_loss'].append(None)
                df['test_acc'].append(None)


            # save model weights
            path = os.path.join(args.output, args.experiment)
            os.makedirs(path, exist_ok=True)
            torch.save(self.model.state_dict(), os.path.join(path, 'W.pt'))

            pd_df = pd.DataFrame.from_dict(df, orient='columns')
            pd_df.to_csv(os.path.join(path, 'log.csv'), index=False, float_format='%g')


class VanillaTrainer(object):
    def __init__(self, model, args):
        self.model = model
        self.device = args.device
        self.clip_grad = args.clip_grad
        self.cutmix_beta = args.cutmix_beta
        self.cutmix_prob = args.cutmix_prob
        print("self.cutmix_beta:", self.cutmix_beta)
        print("self.cutmix_prob:", self.cutmix_prob)
        self.label_smoothing = args.label_smoothing
        self.optimizer = args.optimizer
        self.wandb = args.wandb
        if self.wandb:
            wandb.config.update(args, allow_val_change=True)

        if args.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum,
                                       weight_decay=args.weight_decay, nesterov=args.nesterov)
        elif args.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2),
                                        weight_decay=args.weight_decay)
        else:
            raise ValueError(f"No such optimizer: {self.optimizer}")

        if args.scheduler == 'step':
            self.base_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                 milestones=[args.epochs // 2, 3 * args.epochs // 4],
                                                                 gamma=args.gamma)
        elif args.scheduler == 'cosine':
            self.base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs,
                                                                             eta_min=args.min_lr)
        else:
            raise ValueError(f"No such scheduler: {self.scheduler}")

        if args.warmup_epoch:
            self.scheduler = warmup_scheduler.GradualWarmupScheduler(self.optimizer, multiplier=1.,
                                                                     total_epoch=args.warmup_epoch,
                                                                     after_scheduler=self.base_scheduler)
        else:
            self.scheduler = self.base_scheduler

        # scaler
        self.scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

        self.epochs = args.epochs
        # self.criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        self.criterion = nn.CrossEntropyLoss()

        self.num_steps = 0
        self.epoch_loss, self.epoch_corr, self.epoch_acc = 0., 0., 0.

    def _train_one_step(self, batch, mixup_fn):
        self.model.train()
        img, label = batch
        self.num_steps += 1
        img, label = img.to(self.device), label.to(self.device)
        img, label = mixup_fn(img, label)

        self.optimizer.zero_grad()

        # compute output
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                out = self.model(img)
                loss = self.criterion(out, label)
            self.scaler.scale(loss).backward()
        else:
            out = self.model(img)
            loss = self.criterion(out, label)
            loss.backward()

        if self.clip_grad:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        acc = out.argmax(dim=-1).eq(label.argmax(dim=-1)).sum(-1) / img.size(0)

        self.epoch_tr_loss += loss * img.size(0)
        self.epoch_tr_corr += out.argmax(dim=-1).eq(label.argmax(dim=-1)).sum(-1)

    # @torch.no_grad
    def _test_one_step(self, batch):
        self.model.eval()
        img, label = batch
        img, label = img.to(self.device), label.to(self.device)

        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    out = self.model(img)
                    loss = self.criterion(out, label)
        else:
            with torch.no_grad():
                out = self.model(img)
                loss = self.criterion(out, label)

        self.epoch_loss += loss * img.size(0)
        self.epoch_corr += out.argmax(dim=-1).eq(label).sum(-1)

    def fit(self, train_dl, valid_dl, test_dl, args):
        df = {'epoch': [], 'train_loss': [], 'valid_loss': [], 'train_acc': [], 'valid_acc': [], 'mmc': []}
        mixup_fn = Mixup(
            cutmix_alpha=self.cutmix_beta,
            prob=self.cutmix_prob,
            label_smoothing=self.label_smoothing,
            num_classes=args.num_classes,
        )
        for epoch in trange(1, self.epochs + 1):
            num_tr_imgs = 0.
            self.epoch_tr_loss, self.epoch_tr_corr, self.epoch_tr_acc = 0., 0., 0.
            for batch_idx, batch in enumerate(train_dl):
                self._train_one_step(batch, mixup_fn)
                num_tr_imgs += batch[0].size(0)
            self.epoch_tr_loss /= num_tr_imgs
            self.epoch_tr_acc = self.epoch_tr_corr / num_tr_imgs

            self.scheduler.step()

            df['epoch'].append(epoch)
            df['train_acc'].append(self.epoch_tr_acc.item())
            df['train_loss'].append(self.epoch_tr_loss.item())
            df['mmc'].append(self.model.mmc().item())

            if self.wandb:
                wandb.log({
                    'epoch_tr_loss': self.epoch_tr_loss,
                    'epoch_tr_acc': self.epoch_tr_acc,
                    'epoch_mmc': self.model.mmc().item(),
                }, step=epoch
                )

            num_imgs = 0.
            self.epoch_loss, self.epoch_corr, self.epoch_acc = 0., 0., 0.
            for batch_idx, batch in enumerate(test_dl):
                self._test_one_step(batch)
                num_imgs += batch[0].size(0)
            self.epoch_loss /= num_imgs
            self.epoch_acc = self.epoch_corr / num_imgs

            df['valid_loss'].append(self.epoch_loss.item())
            df['valid_acc'].append(self.epoch_acc.item())

            if self.wandb:
                wandb.log({
                    'val_loss': self.epoch_loss,
                    'val_acc': self.epoch_acc
                }, step=epoch
                )

            # save model weights
            path = os.path.join(args.output, args.experiment)
            os.makedirs(path, exist_ok=True)
            torch.save(self.model.state_dict(), os.path.join(path, 'W.pt'))

            pd_df = pd.DataFrame.from_dict(df, orient='columns')
            pd_df.to_csv(os.path.join(path, f'log.csv'), index=False, float_format='%g')
