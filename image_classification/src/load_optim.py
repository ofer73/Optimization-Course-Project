"""
Load the desired optimizer.
"""

import torch.optim as optim
from sgd_lr_decay import SGDLRDecay
from sls import Sls


def load_optim(params, optim_method, eta0, alpha, c, milestones, T_max,
               n_batches_per_epoch, nesterov, momentum, weight_decay, warmup_steps, tail_steps, restarts_num):
    """
    Args:
        params: iterable of parameters to optimize or dicts defining
            parameter groups.
        optim_method: which optimizer to use.
        eta0: starting step size.
        alpha: decaying factor for various methods.
        c: used in line search.
        milestones: used for SGD stage decay denoting when to decrease the
            step size, unit in iteration.
        T_max: total number of steps.
        n_batches_per_epoch: number of batches in one train epoch.
        nesterov: whether to use nesterov momentum (True) or not (False).
        momentum: momentum factor used in variants of SGD.
        weight_decay: weight decay factor.
        warmup_steps: number of steps using warm  up scheme
        tail_steps: number of steps using tail  scheme
        restarts_num: number of restarts used in cosine annealing scheme


    Outputs:
        an optimizer
    """

    if optim_method == 'SGD' or optim_method == 'SGD_ReduceLROnPlateau':
        optimizer = optim.SGD(params=params, lr=eta0, momentum=momentum,
                              weight_decay=weight_decay, nesterov=nesterov)
    elif optim_method == 'Adam':
        optimizer = optim.Adam(params=params, lr=eta0,
                               weight_decay=weight_decay)
    elif optim_method.startswith('SGD') and optim_method.endswith('Decay'):
        if optim_method == 'SGD_Exp_Decay':
            scheme = 'exp'
        elif optim_method == 'SGD_1t_Decay':
            scheme = '1t'
        elif optim_method == 'SGD_1sqrt_Decay':
            scheme = '1sqrt'
        elif optim_method == 'SGD_Stage_Decay':
            scheme = 'stage'
        elif optim_method == 'SGD_Cosine_Decay':
            scheme = 'cosine'
        elif optim_method == 'SGD_Linear_Decay':
            scheme = 'linear'
        elif optim_method == 'SGD_Linear+w_Decay':
            scheme = 'linear+w'
        elif optim_method == 'SGD_Cosine+w_Decay':
            scheme = 'cosine+w'
        elif optim_method == 'SGD_Linear_Start_Cosine_Tail_Decay':
            scheme = 'linear_start_cosine_tail'
        elif optim_method == 'SGD_Cosine_Start_Linear_Tail_Decay':
            scheme = 'cosine+linear_tail'
        elif optim_method == 'SGD_Exp_Start_Cosine_Tail_Decay':
            scheme = 'exp+cosine_tail'
        elif optim_method == 'SGD_Linear_Start_Exp_Tail_Decay':
            scheme = 'linear_start_exp_tail'
        elif optim_method == 'SGD_Cosine_Annealing_Decay':
            scheme = 'cosine_annealing'
        elif optim_method == 'SGD_Cosine_Eps_Stop_Decay':
            scheme = 'cosine_eps_stop'
        elif optim_method == 'SGD_Cosine_Tail_Curved_Decay':
            scheme = 'cosine_tail_curved'
            
        optimizer = SGDLRDecay(params=params, scheme=scheme, eta0=eta0,
                               alpha=alpha, milestones=milestones, T_max=T_max,
                               momentum=momentum, weight_decay=weight_decay,
                               nesterov=nesterov, warmup_steps=warmup_steps,
                               tail_steps=tail_steps, restarts_num = restarts_num)
    elif optim_method == 'SLS-Armijo0':
        optimizer = Sls(params=params, n_batches_per_epoch=n_batches_per_epoch,
                        init_step_size=eta0, c=c, reset_option=0,
                        line_search_fn="armijo")
    elif optim_method == 'SLS-Armijo1':
        optimizer = Sls(params=params, n_batches_per_epoch=n_batches_per_epoch,
                        init_step_size=eta0, c=c, reset_option=1,
                        line_search_fn="armijo")
    elif optim_method == 'SLS-Armijo2':
        optimizer = Sls(params=params, n_batches_per_epoch=n_batches_per_epoch,
                        init_step_size=eta0, c=c, reset_option=2,
                        line_search_fn="armijo")
    else:
        raise ValueError("Invalid optimizer: {}".format(optim_method))

    return optimizer
