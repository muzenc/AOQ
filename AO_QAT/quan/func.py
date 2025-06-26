from typing import Union
import torch
import torch as t
from torch.nn import functional as F
from torch import Tensor
from torch.nn.common_types import _size_2_t
import matplotlib.pyplot as plt


class QuanConv2d(t.nn.Conv2d):
    def __init__(self, m: t.nn.Conv2d, quan_w_fn=None, quan_a_fn=None):
        assert type(m) == t.nn.Conv2d
        super().__init__(
            m.in_channels,
            m.out_channels,
            m.kernel_size,
            stride=m.stride,
            padding=m.padding,
            dilation=m.dilation,
            groups=m.groups,
            bias=True if m.bias is not None else False,
            padding_mode=m.padding_mode,
        )
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn

        self.weight = t.nn.Parameter(m.weight.detach())
        self.quan_w_fn.init_from(m.weight.detach())
        if m.bias is not None:
            self.bias = t.nn.Parameter(m.bias.detach())

    def forward(self, x: Tensor) -> Tensor:
        quantized_weight = self.quan_w_fn(self.weight)
        quantized_act = self.quan_a_fn(x)
        out = self._conv_forward(quantized_act, quantized_weight, bias=self.bias)

        # fname = "act.png"
        # plt.figure(3)
        # plt.clf()
        # plt.hist(
        #     act.reshape(-1).cpu().detach().numpy(),
        #     bins=200,
        #     range=(-10.0, 10.0),
        # )
        # plt.savefig(fname)
        return out


class QuanLinear(t.nn.Linear):
    def __init__(self, m: t.nn.Linear, quan_w_fn=None, quan_a_fn=None):
        assert type(m) == t.nn.Linear
        super().__init__(m.in_features, m.out_features, bias=True if m.bias is not None else False)
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn

        self.weight = t.nn.Parameter(m.weight.detach())
        self.quan_w_fn.init_from(m.weight.detach())
        if m.bias is not None:
            self.bias = t.nn.Parameter(m.bias.detach())

    def forward(self, x: Tensor) -> Tensor:
        quantized_weight = self.quan_w_fn(self.weight)
        quantized_act = self.quan_a_fn(x)
        return F.linear(quantized_act, quantized_weight, bias=self.bias)


# class QuanBN(t.nn.BatchNorm2d):
#     def __init__(self, m: t.nn.BatchNorm2d, quan_a_fn=None):
#         assert type(m) == t.nn.BatchNorm2d
#         super(QuanBN, self).__init__(m.num_features)
#         self.weight = torch.nn.Parameter(m.weight.detach())
#         self.bias = torch.nn.Parameter(m.bias.detach())
#         self.running_mean = m.running_mean
#         self.running_var = m.running_var

#         self.flag = 0
#         self.quan_a_fn = quan_a_fn

#     def forward(self, input: Tensor) -> Tensor:
#         out = super(QuanBN, self).forward(input)
#         if self.flag == 0:
#             self.quan_a_fn.init_from(out)
#             self.flag = 1
#         quantized_act = self.quan_a_fn(out)
#         return quantized_act
#         # return out


QuanModuleMapping = {
    t.nn.Conv2d: QuanConv2d,
    t.nn.Linear: QuanLinear,
}
