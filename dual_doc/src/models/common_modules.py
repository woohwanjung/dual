from collections import OrderedDict

import torch
from torch import nn
import numpy as np




class PredictionBiLinear(nn.Module):
    def __init__(self, dim_in1, dim_in2, dim_out, bias = True):
        super().__init__()
        self.bili = nn.Bilinear(dim_in1,dim_in2,dim_out, bias = bias)

    def forward(self, input1,  input2):
        output = self.bili(input1, input2)
        return output

    def add_bias(self, bias, requires_grad = True):
        if isinstance(bias, np.ndarray):
            bias = torch.tensor(bias, device = self.bili.bias.device, dtype = self.bili.bias.dtype)
        bias_after = self.bili.bias + bias
        self.fix_bias(bias_after, requires_grad = requires_grad)

    def fix_bias(self, bias, requires_grad = False):
        if isinstance(bias, np.ndarray):
            bias = torch.tensor(bias, device = self.bili.bias.device, dtype = self.bili.bias.dtype)
        bias_dict = OrderedDict({'bias': bias})
        self.bili.load_state_dict(bias_dict, strict=False)
        self.bili.bias.requires_grad = requires_grad

class MultiLayerPerceptron(nn.Module):
    def __init__(self, dim_in, dim_out, hidden_dims = [], activation = "tanh"):
        super().__init__()

        dim_in_list = [dim_in ] + hidden_dims
        dim_out_list = hidden_dims + [dim_out]
        linear_layers = [ nn.Linear(d_in, d_out) for d_in, d_out in zip(dim_in_list, dim_out_list)]
        self.linear_layers = nn.ModuleList(linear_layers)

        if activation == "tanh":
            self.activation = torch.nn.Tanh()
        elif activation == "relu":
            self.activation = torch.nn.ReLU()

    def forward(self, vect_in):
        for l in range (len(self.linear_layers)-1):
            vect_out = self.linear_layers[l](vect_in)
            vect_in = self.activation(vect_out)

        vect_out = self.linear_layers[-1](vect_in)
        return vect_out



class PredictionLinear(nn.Module):
    def __init__(self, dim_in, dim_out, bias = True):
        super().__init__()
        self.li = nn.Linear(dim_in, dim_out, bias = bias)

    def forward(self, input1):
        output = self.li(input1)
        return output

    def add_bias(self, bias, requires_grad = True):
        if isinstance(bias, np.ndarray):
            bias = torch.tensor(bias, device = self.li.bias.device, dtype = self.li.bias.dtype)
        bias_after = self.li.bias + bias
        self.fix_bias(bias_after, requires_grad = requires_grad)

    def fix_bias(self, bias, requires_grad = False):
        if isinstance(bias, np.ndarray):
            bias = torch.tensor(bias, device = self.li.bias.device, dtype = self.li.bias.dtype)
        bias_dict = OrderedDict({'bias': bias})
        self.li.load_state_dict(bias_dict, strict=False)
        self.li.bias.requires_grad = requires_grad


class PredictionBiLinearMulti(nn.Module):
    def __init__(self, num_output_modules, dim_in1, dim_in2, dim_out, bias = True):
        super().__init__()
        self.dim_out = dim_out
        self.num_output_modules = num_output_modules
        self.bili = nn.Bilinear(dim_in1,dim_in2,dim_out * self.num_output_modules, bias = bias)

    def forward(self, input1,  input2):
        output = self.bili(input1, input2)
        output_list = [output[:,:,i*self.dim_out:(i+1)*self.dim_out] for i in range(self.num_output_modules)]
        return output_list

    def twin_init(self):
        self.bili.weight[self.dim_out:2*self.dim_out] = self.bili.weight[:self.dim_out]
        self.bili.bias[self.dim_out:2*self.dim_out] = self.bili.bias[:self.dim_out]
        pass

if __name__ == "__main__":
    bili_mul = PredictionBiLinearMulti(2,3,3,4)
    bili_mul.twin_init()
    '''
    net = BiLinearOutput(5,2,3)
    bias = np.ones(3)
    bias[2:]+=1

    net.fix_bias(bias)
    '''