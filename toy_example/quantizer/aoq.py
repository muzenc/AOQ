import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from globalVal import globalVal
from .gmm.gmm_fit import gmm_fit
import math
import numpy as np

device = globalVal.device


class QuantizedLinear_AOQ(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(QuantizedLinear_AOQ, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features)), requires_grad=True)
        self.step_size = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.register_buffer("level", torch.tensor([0.0] * 4))
        self.register_buffer("nodes", torch.tensor([0.0] * 4))
        self.one = torch.tensor([1.0], device=device)
        self.alpha = 1.0
        self.iter = 0.0
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def round(self):
        x = self.weight
        step_size = self.step_size.clone().detach()
        levels = torch.cat(
            (
                -2.0 * step_size,
                -1.0 * step_size,
                0.0 * step_size,
                1.0 * step_size,
            )
        )
        nodes = torch.cat((-1.5 * step_size, -0.5 * step_size, 0.5 * step_size))
        for i in range(3):
            if i == 0:
                step_right = levels[i + 1]
                x_forward = torch.where(x >= nodes[0], step_right, levels[0])
            else:
                step_right = levels[i + 1]
                x_forward = torch.where(x >= nodes[i], step_right, x_forward)
        self.weight.data = x_forward

    def forward(self, input):
        # self.alpha = 0.4
        x = self.weight
        if globalVal.epoch <= 50 and globalVal.epoch % 5 == 0:
            self.alpha = 0.35 * np.cos(2 * np.pi * globalVal.epoch / 100) + 0.65
        elif globalVal.epoch >= 85 and globalVal.epoch <= 100 and globalVal.epoch % 5 == 0:
            self.alpha = 0.35 * np.cos(2 * np.pi * (globalVal.epoch - 25) / 100) + 0.65
        self.level = torch.cat(
            (
                -2.0 * self.step_size * self.alpha,
                -1.0 * self.step_size * self.alpha,
                0.0 * self.step_size * self.alpha,
                1.0 * self.step_size * self.alpha,
            )
        )
        if globalVal.epoch <= 150:
            self.level = torch.cat(
                (
                    -2.0 * self.one * self.alpha,
                    -1.0 * self.one * self.alpha,
                    0.0 * self.one * self.alpha,
                    1.0 * self.one * self.alpha,
                )
            )
        # if globalVal.epoch <= 200 and globalVal.epoch % 10 == 0:
        #     self.alpha = 0.4 * np.cos(2 * np.pi * globalVal.epoch / 200) + 0.6
        # self.alpha = 1.0
        globalVal.alpha = self.alpha
        self.nodes = torch.cat(
            (
                -1.5 * self.alpha * self.one,
                -0.5 * self.alpha * self.one,
                0.5 * self.alpha * self.one,
            )
        )
        cluster = torch.cat(
            (
                -2.0 * self.alpha * self.one,
                -1.0 * self.alpha * self.one,
                0.0 * self.alpha * self.one,
                1.0 * self.alpha * self.one,
            )
        )
        # self.nodes = torch.cat(
        #     (
        #         torch.tensor([-0.6], device=device),
        #         torch.tensor([-0.2], device=device),
        #         torch.tensor([0.2], device=device),
        #     )
        # )
        # level_copy = self.level.clone().detach()
        # if self.iter % 505 == 0:
        #     self.nodes = gmm_fit(x.clone().detach(), level_copy)
        # self.iter += 1.0
        for i in range(3):
            if i == 0:
                step_right = self.level[i + 1]
                x_forward = torch.where(x >= self.nodes[0], step_right, self.level[0])
                x_cluster = torch.where(x >= self.nodes[0], cluster[i + 1], cluster[0])
            else:
                step_right = self.level[i + 1]
                x_forward = torch.where(x >= self.nodes[i], step_right, x_forward)
                x_cluster = torch.where(x >= self.nodes[i], cluster[i + 1], x_cluster)

        # mask = x < -2.5 * self.alpha * self.one
        # self.weight.data[mask] = -2.0 * self.alpha * self.one
        # mask = x > 1.5 * self.alpha * self.alpha * self.one
        # self.weight.data[mask] = 1.0 * self.alpha * self.one
        # mask = x < -2.5 * self.alpha * self.step_size
        # self.weight.data[mask] = self.weight.data[mask] + self.alpha * self.step_size
        # mask = x > 1.5 * self.alpha * self.step_size
        # self.weight.data[mask] = self.weight.data[mask] - self.alpha * self.step_size
        x_backward = 1.0 * x_forward + 1.0 * x
        out = x_backward + (x_forward - x_backward).detach()
        globalVal.loss = torch.norm(self.weight - x_cluster.detach())
        return F.linear(input, out, self.bias)
