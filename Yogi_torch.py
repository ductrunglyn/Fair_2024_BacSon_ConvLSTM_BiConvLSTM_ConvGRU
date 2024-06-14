import torch
from torch.optim.optimizer import Optimizer
import tensorflow_addons as tfa

class Yogi(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-3):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(Yogi, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Yogi does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['mu'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)

                # Get current state
                mu = state['mu']
                v = state['v']
                beta1, beta2 = group['betas']
                state['step'] += 1
                step = state['step']

                # Update momentum and velocity
                mu = beta1 * mu + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad ** 2

                # Bias correction
                mu_hat = mu / (1 - beta1 ** step)
                v_hat = v / (1 - beta2 ** step)

                # Update parameters
                p.data -= group['lr'] * mu_hat / (torch.sqrt(v_hat) + group['eps'])

                # Save state
                state['mu'] = mu
                state['v'] = v

        return loss
