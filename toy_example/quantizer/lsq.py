import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from globalVal import globalVal

device = globalVal.device


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


class QuantizedLinear_LSQ(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(QuantizedLinear_LSQ, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.step_size = nn.Parameter(torch.tensor(1.8), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=0)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        thd_neg, thd_pos = -2.0, 1.0
        # print("step_size", self.step_size)
        step_size = self.step_size.clone().detach()
        with torch.no_grad():
            self.weight.data = torch.where(
                self.weight <= (thd_neg) * step_size,
                (thd_neg + 0.2) * step_size,
                self.weight.data,
            )
            self.weight.data = torch.where(
                self.weight >= (thd_pos) * step_size,
                (thd_pos - 0.2) * step_size,
                self.weight.data,
            )
        x = self.weight / self.step_size
        x = torch.clamp(x, thd_neg, thd_pos)
        x = round_pass(x)
        q_weight = x * self.step_size
        globalVal.loss = torch.norm(self.weight - q_weight.detach())
        return F.linear(input, q_weight, self.bias)
