"""
Adapted from https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py
"""
import math

import numpy as np
import torch
from torch.optim import Optimizer

class SGDLRDecay(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum)
    with several step size decay schemes (note that t starts from 1):
        1. 1/t decay: eta_t = eta_0 / (1 + alpha*t);
        2. 1/sqrt(t) decay: eta_t = eta_0 / (1 + alpha*sqrt(t));
        3. exponential decay: eta_t = eta_0 * (alpha**t);
        4. stagewise sgd: multiply eta_t by alpha at each milestone.
        5. cosine decay: eta_t = 0.5 * (1 + cos(t*pi/T)) * eta_0

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        scheme (str): the decay scheme, currently only supports {'exp', '1t',
            '1sqrt', 'stage'}.
        eta0 (float): initial learning rate.
        alpha (float): decay factor.
        milestones (list): a list denoting which time to decrease the stepsize.
        T_max: total number of steps.
        momentum (float, optional): momentum factor (default: 0).
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0).
        dampening (float, optional): dampening for momentum (default: 0).
        nesterov (bool, optional): enables Nesterov momentum (default: False).
    """

    def __init__(self, params, scheme, eta0, alpha, milestones=[], T_max=0,
                 momentum=0, dampening=0, weight_decay=0, nesterov=False, warmup_steps=0, tail_steps=0, restarts_num=1):
        if eta0 < 0.0:
            raise ValueError("Invalid eta0 value: {}".format(eta0))
        if alpha < 0.0 and scheme != 'cosine_tail_curved':
            raise ValueError("Invalid alpha value: {}".format(alpha))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight decay: {}".format(weight_decay))

        defaults = dict(momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDLRDecay, self).__init__(params, defaults)

        self.eta0 = eta0
        self.alpha = alpha
        self.milestones = [int(x) for x in milestones]
        self.cur_round = 0
        self.cur_lr = eta0
        self.T_max = T_max
        self.warmup_steps = warmup_steps
        self.tail_steps = tail_steps
        self.restarts_num = restarts_num

        # Define the function for computing the current step size for each decay.
        self.get_lr_func = None
        if scheme == 'exp':
            self.get_lr_func = lambda cur_lr, t, eta0, alpha, milestones, T_max: cur_lr * alpha
        elif scheme == '1t':
            self.get_lr_func = lambda cur_lr, t, eta0, alpha, milestones, T_max: eta0 / (1.0 + alpha*t)
        elif scheme == '1sqrt':
            self.get_lr_func = lambda cur_lr, t, eta0, alpha, milestones, T_max: eta0 / (1.0 + alpha*(t**0.5))
        elif scheme == 'stage':
            self.get_lr_func = lambda cur_lr, t, eta0, alpha, milestones, T_max: cur_lr * alpha if t in milestones else cur_lr
        elif scheme == 'cosine':
            self.get_lr_func = lambda cur_lr, t, eta0, alpha, milestones, T_max: 0.5 * (1 + math.cos(t*math.pi/T_max)) * eta0
        elif scheme == 'cosine_annealing':
            T_0 = self.T_max // self.restarts_num
            self.get_lr_func = lambda cur_lr, t, eta0, alpha, milestones, T_max: self.cosine_annealing(T_0, eta0)
        elif scheme == 'linear':
            self.get_lr_func = lambda cur_lr, t, eta0, alpha, milestones, T_max, end_lr=0 : eta0 * ((1 - float(t) / T_max))
        # For the warm-up up scheduelers, eta 0 will be the global maximum of learning rates
        # (reached in last warm-up step)
        elif scheme == 'linear+w':
            self.get_lr_func = lambda cur_lr, t, eta0, alpha, milestones, T_max, end_lr=0 : eta0 * ((1 - float(t) / T_max)) \
                if t > self.warmup_steps else eta0 * (t / self.warmup_steps)
        elif scheme == 'cosine+w':
            turn_t = T_max - self.warmup_steps
            const = (eta0 * (turn_t / self.warmup_steps)) / (
                    0.5 * (1 + math.cos(turn_t * math.pi / T_max)) * eta0) if self.warmup_steps > 0 else 1
            self.get_lr_func = lambda cur_lr, t, eta0, alpha, milestones, T_max: 0.5 * (1 + math.cos(t*math.pi/T_max)) * eta0 * const \
                if t > self.warmup_steps else eta0 * (t / self.warmup_steps)
        elif scheme == 'linear_start_cosine_tail':
            turn_t = T_max - self.tail_steps
            const = (eta0 * ((1 - float(turn_t) / T_max))) / (
                        0.5 * (1 + math.cos(turn_t * math.pi / T_max)) * eta0) if self.tail_steps > 0 else 1
            self.get_lr_func = lambda cur_lr, t, eta0, alpha, milestones, T_max: 0.5 * (1 + math.cos(t * math.pi / T_max)) * eta0 * const \
                if t > T_max - self.tail_steps else eta0 * ((1 - float(t) / T_max))
        elif scheme == 'linear_start_exp_tail':
            self.get_lr_func = lambda cur_lr, t, eta0, alpha, milestones, T_max: cur_lr * alpha \
                if t > T_max - self.tail_steps else eta0 * ((1 - float(t) / T_max))
        elif scheme == "cosine+linear_tail":
            self.get_lr_func = lambda cur_lr, t, eta0, alpha, milestones, T_max: (0.5*(1+math.cos((T_max-self.tail_steps)*math.pi/T_max))*eta0) - ((t-T_max+self.tail_steps)*(0.5*(1+math.cos((T_max-self.tail_steps)*math.pi/T_max))*eta0)/self.tail_steps) \
                if t > T_max - self.tail_steps else 0.5 * (1 + math.cos(t*math.pi/T_max)) * eta0
        elif scheme == "exp+cosine_tail":
            self.get_lr_func = lambda cur_lr, t, eta0, alpha, milestones, T_max: (eta0 * (alpha**t))  /  (1 + math.cos((T_max - self.tail_steps) * math.pi / T_max)) * (1 + math.cos(t * math.pi / T_max)) \
                if t > T_max - self.tail_steps else cur_lr * alpha
        elif scheme == 'cosine_eps_stop':
            self.get_lr_func = lambda cur_lr, t, eta0, alpha, milestones, T_max: 0.5 * (1 + math.cos(t*math.pi/T_max)) * eta0*(1-alpha) + eta0*alpha
        elif scheme == 'cosine_tail_curved':
            self.get_lr_func = lambda cur_lr, t, eta0, alpha, milestones, T_max: eta0*(1+math.cos(t*math.pi/T_max))/2 if t < T_max/2 \
                else ((1+alpha)*(eta0*T_max**2-T_max*eta0*t)+(eta0**2-alpha*T_max**2)*(eta0*(1+math.cos(t*math.pi/T_max))/2))/(T_max**2+eta0**2)



    def __setstate__(self, state):
        super(SGDLRDecay, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def cosine_annealing(self, T_max, eta0):
        # if we reached the restart milestone, restart curr_round to 0
        if self.cur_round == T_max + 1:
            self.cur_round = 0
        return 0.5 * (1 + math.cos(self.cur_round*math.pi/T_max)) * eta0

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        self.cur_round += 1

        self.cur_lr = self.get_lr_func(self.cur_lr, self.cur_round, self.eta0,
                                       self.alpha, self.milestones, self.T_max)

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-self.cur_lr, d_p)

        return loss
