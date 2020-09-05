import math
import torch

def adam_step(optimizer, closure=None):
    """From https://pytorch.org/docs/stable/_modules/torch/optim/adam.html#Adam"""
    """Performs a single optimization step.

    Arguments:
        closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
    """
    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()

    w_steps = list()
    adam_grads = list()
    adam_grad_numers = list()
    adam_grad_denoms = list()
    grads = list()

    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
            amsgrad = group['amsgrad']

            state = optimizer.state[p]

            # State initialization
            if len(state) == 0:
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

            exp_avg, exp_avg_sq = state['exp_avg'].clone(), state['exp_avg_sq'].clone()
            if amsgrad:
                max_exp_avg_sq = state['max_exp_avg_sq'].clone()
            beta1, beta2 = group['betas']

            #state['step'] += 1
            step = state['step'] + 1
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            if group['weight_decay'] != 0:
                grad = grad.add(p, alpha=group['weight_decay'])

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
            else:
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

            numer = exp_avg / bias_correction1
            adam_grad = numer / denom
            lr = group['lr']
            w_step = lr * adam_grad
            w_steps.append(w_step.view(-1))
            adam_grads.append(adam_grad.view(-1))
            adam_grad_numers.append(numer.view(-1))
            adam_grad_denoms.append(denom.view(-1))
            grads.append(grad.view(-1))

    w_steps = torch.cat(w_steps)
    adam_grads = torch.cat(adam_grads)
    adam_grad_numers = torch.cat(adam_grad_numers)
    adam_grad_denoms = torch.cat(adam_grad_denoms)
    grads = torch.cat(grads)

    w_step_abs_mean = torch.mean(torch.abs(w_steps))
    adam_grad_abs_mean = torch.mean(torch.abs(adam_grads))
    adam_grad_numers_abs_mean = torch.mean(torch.abs(adam_grad_numers))
    adam_grad_denoms_abs_mean = torch.mean(torch.abs(adam_grad_denoms))
    grad_abs_mean = torch.mean(torch.abs(grads))

    return w_step_abs_mean, adam_grad_abs_mean, adam_grad_numers_abs_mean, \
        adam_grad_denoms_abs_mean, grad_abs_mean
