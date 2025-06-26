import torch
import torch.nn as nn
import sys
from matplotlib import pyplot as plt
import numpy as np

sys.path.append("..")
from globalVal import globalVal


device = torch.device(globalVal.device)


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


class LTQ(nn.Module):
    def __init__(self, num_bits, quantized_object):
        super(LTQ, self).__init__()
        init_range = 2.0
        self.num_bits = num_bits
        self.register_buffer("n_val", torch.tensor([2**num_bits - 1]))
        self.register_buffer("interval", torch.tensor([init_range / self.n_val]))
        self.register_buffer("zero", torch.tensor([0.0]))
        self.register_buffer("two", torch.tensor([2.0]))
        self.quantized_object = quantized_object
        self.eps = nn.Parameter(torch.tensor([1e-6]), requires_grad=False)

        self.start = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.register_buffer("start_detach", torch.tensor([0.0]))
        self.a = nn.Parameter(torch.tensor([self.interval] * self.n_val), requires_grad=True)
        self.scale1 = nn.Parameter(torch.tensor([1.0]))
        self.scale2 = nn.Parameter(torch.tensor([2.0]))

    def forward(self, x):
        if self.quantized_object == "act":
            x = x * self.scale1

            self.start_detach = self.start.clone().detach()
            a_pos = torch.where(self.a > self.eps, self.a, self.eps)
            step_right = self.zero + 0.0
            for i in range(self.n_val):
                step_right += self.interval
                if i == 0:
                    thre_forward = self.start + a_pos[0] / 2
                    thre_backward = self.start + 0.0
                    x_forward = torch.where(x > thre_forward, step_right, self.zero)
                    x_backward = torch.where(
                        x > thre_backward,
                        self.interval / a_pos[i] * (x - thre_backward) + step_right - self.interval,
                        self.zero,
                    )
                else:
                    thre_forward += a_pos[i - 1] / 2 + a_pos[i] / 2
                    thre_backward += a_pos[i - 1]
                    x_forward = torch.where(x > thre_forward, step_right, x_forward)
                    x_backward = torch.where(
                        x > thre_backward,
                        self.interval / a_pos[i] * (x - thre_backward) + step_right - self.interval,
                        x_backward,
                    )

            thre_backward += a_pos[i]
            x_backward = torch.where(x > thre_backward, self.two, x_backward)
            out = x_forward.detach() + x_backward - x_backward.detach()
            out = out * self.scale2
        return out


class AOQ(nn.Module):
    def __init__(self, num_bits, quantized_object):
        super(AOQ, self).__init__()
        self.step_size = nn.Parameter(torch.tensor([1.0]))
        self.num_bits = num_bits
        self.register_buffer("level", torch.tensor([0.0] * (2**num_bits)))
        self.register_buffer("threshold", torch.tensor([0.0] * (2**num_bits - 1)))
        self.alpha = nn.Parameter(torch.tensor([1.0], device=device), requires_grad=False)
        self.init_interval = nn.Parameter(torch.tensor([1.0]), requires_grad=False)

    def init_from(self, x):
        self.step_size.data = torch.tensor([torch.std(x.clone().detach()) / (2 ** (self.num_bits - 2))])
        self.init_interval.data = torch.tensor([torch.std(x.clone().detach()) / (2 ** (self.num_bits - 2))])

    def forward(self, x):
        if globalVal.epoch <= 50 and globalVal.epoch % 5 == 0:
            self.alpha.data = torch.tensor(
                [0.35 * np.cos(2 * np.pi * globalVal.epoch / 100) + 0.65],
                dtype=torch.float32,
                device=device,
            )
        
        if globalVal.epoch <= 50:
            self.level = torch.cat(
                (
                    -1.5 * self.init_interval * self.alpha,
                    -0.5 * self.init_interval * self.alpha,
                    0.5 * self.init_interval * self.alpha,
                    1.5 * self.init_interval * self.alpha,
                )
            )
        elif globalVal.epoch > 50:
            self.level = torch.cat(
                (
                    -1.5 * self.step_size * self.alpha,
                    -0.5 * self.step_size * self.alpha,
                    0.5 * self.step_size * self.alpha,
                    1.5 * self.step_size * self.alpha,
                )
            )
        self.threshold = torch.cat(
            (
                -1.0 * self.init_interval * self.alpha,
                0.0 * self.init_interval * self.alpha,
                1.0 * self.init_interval * self.alpha,
            )
        )
        cluster = torch.cat(
            (
                -1.5 * self.init_interval * self.alpha,
                -0.5 * self.init_interval * self.alpha,
                0.5 * self.init_interval * self.alpha,
                1.5 * self.init_interval * self.alpha,
            )
        )
        for i in range(2**self.num_bits - 1):
            if i == 0:
                x_forward = torch.where(x >= self.threshold[0], self.level[i + 1], self.level[0])
                x_cluster = torch.where(x >= self.threshold[0], cluster[i + 1], cluster[0])
            else:
                x_forward = torch.where(x >= self.threshold[i], self.level[i + 1], x_forward)
                x_cluster = torch.where(x >= self.threshold[i], cluster[i + 1], x_cluster)
        x_backward = 1.0 * x_forward + 1.0 * x
        out = x_backward + (x_forward - x_backward).detach()
        globalVal.loss += torch.norm(x - x_cluster.detach())
        return out


class LSQ(nn.Module):
    def __init__(self, num_bits, quantized_object):
        super(LSQ, self).__init__()
        self.step_size = nn.Parameter(torch.tensor([1.0]))
        self.num_bits = num_bits
        if quantized_object == "act":
            self.thd_neg = 0
            self.thd_pos = 3
        elif quantized_object == "weight":
            self.thd_neg = -2
            self.thd_pos = 1

    def init_from(self, x):
        self.step_size.data = torch.tensor(x.detach().abs().mean() * 2 / (self.thd_pos**0.5))

    def forward(self, x):
        x_temp = x / self.step_size
        x_temp = torch.clamp(x_temp, self.thd_neg, self.thd_pos)
        x_temp = round_pass(x_temp)
        q_x = x_temp * self.step_size
        return q_x
