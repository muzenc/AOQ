from .func import *
from .quantizer import *
import torch.nn as nn


def find_modules_to_quantize(model, n_bit):
    replaced_modules = dict()
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and name != "conv1":
            replaced_modules[name] = QuanModuleMapping[type(module)](
                module,
                quan_w_fn=AOQ(num_bits=n_bit, quantized_object="weight"),
                quan_a_fn=LTQ(num_bits=n_bit, quantized_object="act"),
            )
    return replaced_modules


def replace_module_by_names(model, modules_to_replace):
    def helper(child: t.nn.Module):
        for n, c in child.named_children():
            if type(c) in QuanModuleMapping.keys():
                for full_name, m in model.named_modules():
                    if c is m and full_name in modules_to_replace.keys():
                        child.add_module(n, modules_to_replace.pop(full_name))
                        break
            else:
                helper(c)

    helper(model)
    return model
