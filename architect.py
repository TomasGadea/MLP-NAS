import torch.nn.functional as F
import torch
import copy


class Architect():
    """ Compute gradients of alphas """

    def __init__(self, net, w_momentum, w_weight_decay, a_optimizer, use_amp=False):
        """
        Args:
            net
            w_momentum: weights momentum
        """
        self.net = net
        # print("self.net = ", net)
        # print("self.net.named_params() = ", [n for n,p in self.net.named_parameters()])
        self.v_net = copy.deepcopy(net)
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay
        self.a_optimizer = a_optimizer
        # scaler
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None

    def virtual_step(self, trn_X, trn_y, xi, w_optim):
        """
        Compute unrolled weight w' (virtual step)
        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient
        Args:
            xi: learning rate for virtual gradient step (same as weights lr)
            w_optim: weights optimizer
        """
        #with torch.cuda.amp.autocast():
        # forward & calc loss
        loss = self.net.loss(trn_X, trn_y)  # L_trn(w)

        # compute gradient
        gradients = torch.autograd.grad(loss, self.net.weights())

        # do virtual step (update gradient)
        # below operations do not need gradient tracking
        with torch.no_grad():
            # dict key is not the value, but the pointer. So original network weight have to
            # be iterated also.
            for w, vw, g in zip(self.net.weights(), self.v_net.weights(), gradients):
                m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
                vw.copy_(w - xi * (m + g + self.w_weight_decay * w))

            # synchronize alphas
            for a, va in zip(self.net.get_alphas(), self.v_net.get_alphas()):
                va.copy_(a)

    def L_01(self, v_net=True):
        """ Compute L_01 loss
        """
        if v_net:
            b = torch.cat([F.sigmoid(a) for a in self.v_net.get_alphas()], dim=0)
            return -F.mse_loss(b, 0.5 + torch.zeros(b.shape, requires_grad=False).cuda())

        b = torch.cat([F.sigmoid(a) for a in self.net.get_alphas()], dim=0)
        return -F.mse_loss(b, 0.5 + torch.zeros(b.shape, requires_grad=False).cuda())

    def rolled_backward(self, val_X, val_y, w01, l):
        self.a_optimizer.zero_grad()
        # print('alpha_grads: ', self.net.get_alpha_grads())  # should be None
        arch_loss = w01 * self.L_01(v_net=False)  # self.net.loss(val_X, val_y) + w01 * self.L_01(v_net=True) # L_trn(w)
        friction_loss = l * self.net.friction()

        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                net_loss = self.net.loss(val_X, val_y)
                loss = net_loss + arch_loss + friction_loss
            self.scaler.scale(loss).backward()
            self.scaler.step(self.a_optimizer)
            self.scaler.update()
        else:
            net_loss = self.net.loss(val_X, val_y)
            loss = net_loss + arch_loss + friction_loss
            loss.backward()
            self.a_optimizer.step()

        # print('alpha_grads: ', self.net.get_alpha_grads())  # should be ≠ 0 or None

        return loss.item(), arch_loss.item(), net_loss.item(), friction_loss.item()

    def unrolled_backward(self, trn_X, trn_y, val_X, val_y, xi, w_optim, w01):
        """ Compute unrolled loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        # do virtual step (calc w`)
        self.virtual_step(trn_X, trn_y, xi, w_optim)

        # calc unrolled loss
        lll = self.v_net.loss(val_X, val_y)
        loss = lll + w01 * self.L_01()  # L_val(w`)
        # print(f"loss: {lll}, w01 * L01: {w01*self.L_01()}, ratio: {lll / (w01*self.L_01())}")
        # print("alpha loss", loss)

        # compute gradient
        v_alphas = tuple(self.v_net.get_alphas())
        v_weights = tuple(self.v_net.weights())
        v_grads = torch.autograd.grad(loss, v_alphas + v_weights)
        dalpha = v_grads[:len(v_alphas)]
        dw = v_grads[len(v_alphas):]

        hessian = self.compute_hessian(dw, trn_X, trn_y)

        # update final gradient = dalpha - xi*hessian
        with torch.no_grad():
            for alpha, da, h in zip(self.net.get_alphas(), dalpha, hessian):
                alpha.grad = da - xi * h

    def compute_hessian(self, dw, trn_X, trn_y):
        """
        dw = dw' { L_val(w', alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
        eps = 0.01 / ||dw||
        """
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm

        # w+ = w + eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d
        loss = self.net.loss(trn_X, trn_y)
        dalpha_pos = torch.autograd.grad(loss, self.net.get_alphas())  # dalpha { L_trn(w+) }

        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p -= 2. * eps * d
        loss = self.net.loss(trn_X, trn_y)
        dalpha_neg = torch.autograd.grad(loss, self.net.get_alphas())  # dalpha { L_trn(w-) }

        # recover w
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d

        hessian = [(p - n) / 2. * eps for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian