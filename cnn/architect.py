import torch
import numpy as np
import torch.nn as nn


def _concat(xs):
    return torch.cat([x.reshape(-1) for x in xs])


class Architect(object):

    def __init__(self, model, args):
        self.device = args.device
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        loss = self.model._loss(input, target)

        with torch.no_grad():
            theta = _concat([p for p in self.model.parameters()]).clone()

            # 取 momentum_buffer，如果沒有就給零 tensor，並 concat 成單一 tensor
            try:
                moment = _concat([
                    network_optimizer.state[v]['momentum_buffer']
                    if 'momentum_buffer' in network_optimizer.state[v] else torch.zeros_like(v)
                    for v in self.model.parameters()
                ]).mul_(self.network_momentum)
            except Exception:
                moment = torch.zeros_like(theta)

            # 這裡保證是單一 tensor
            dtheta = _concat(torch.autograd.grad(
                loss, self.model.parameters()))
            dtheta = dtheta + self.network_weight_decay * theta

            new_theta = theta.sub(moment + dtheta, alpha=eta)

        unrolled_model = self._construct_model_from_theta(new_theta)
        return unrolled_model

    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(
                input_train, target_train, input_valid, target_valid, eta, network_optimizer)
        else:
            self._backward_step(input_valid, target_valid)
        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid):
        loss = self.model._loss(input_valid, target_valid)
        loss.backward()

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        unrolled_model = self._compute_unrolled_model(
            input_train, target_train, eta, network_optimizer)
        unrolled_loss = unrolled_model._loss(input_valid, target_valid)

        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.detach() for v in unrolled_model.parameters()]
        implicit_grads = self._hessian_vector_product(
            vector, input_train, target_train)

        for g, ig in zip(dalpha, implicit_grads):
            g.sub_(ig, alpha=eta)

        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = g.detach().to(self.device)
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = v.numel()
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.to(self.device)

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        R = r / _concat(vector).norm()
        with torch.no_grad():
            for p, v in zip(self.model.parameters(), vector):
                p.add_(v, alpha=R)
        loss = self.model._loss(input, target)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

        with torch.no_grad():
            for p, v in zip(self.model.parameters(), vector):
                p.sub_(v, alpha=2 * R)
        loss = self.model._loss(input, target)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        with torch.no_grad():
            for p, v in zip(self.model.parameters(), vector):
                p.add_(v, alpha=R)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
