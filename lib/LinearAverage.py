import torch
from torch.autograd import Function
from torch import nn
import math

class LinearAverageOp(Function):
    @staticmethod
    def forward(self, x, y, memory, params):
        T = params[0].item()
        batchSize = x.size(0)

        # inner product
        out = torch.mm(x.data, memory.t())
        out.div_(T) # batchSize * N
        
        self.save_for_backward(x, memory, y, params)

        return out

    @staticmethod
    def backward(self, gradOutput):
        x, memory, y, params = self.saved_tensors
        batchSize = gradOutput.size(0)
        T = params[0].item()
        momentum = params[1].item()
        
        # add temperature
        gradOutput.data.div_(T)

        # gradient of linear
        gradInput = torch.mm(gradOutput.data, memory)
        gradInput.resize_as_(x)

        # update the non-parametric data
        weight_pos = memory.index_select(0, y.data.view(-1)).resize_as_(x)
        weight_pos.mul_(momentum)
        weight_pos.add_(torch.mul(x.data, 1-momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        memory.index_copy_(0, y, updated_weight)
        
        return gradInput, None, None, None

class LinearAverage(nn.Module):

    def __init__(self, inputSize, outputSize, T=0.07, momentum=0.5):
        super(LinearAverage, self).__init__()
        stdv = 1 / math.sqrt(inputSize)
        self.nLem = outputSize

        self.register_buffer('params',torch.tensor([T, momentum]));
        stdv = 1. / math.sqrt(inputSize/3)
        self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2*stdv).add_(-stdv))

    def forward(self, x, y):
        out = LinearAverageOp.apply(x, y, self.memory, self.params)
        return out


class LinearAverageOpContinued(Function):
    @staticmethod
    def forward(self, x, x_index, y, memory, params, inv_map):
        device = 'cuda:5' if torch.cuda.is_available() else 'cpu'
        T = params[0].item()
        batchSize = x.size(0)
        vi = trainFeature_ni = torch.zeros(len(inv_map), x.size(1)).to(device)
        for k,v in inv_map.items():
            vi[k, :] = torch.nn.functional.normalize(memory[v].mean(axis=0),dim=0) 
        # inner product
        out = torch.mm(x.data, vi.t())
        out.div_(T) # batchSize * Ni
        
        self.save_for_backward(x, x_index, memory, vi, y, params)

        return out

    @staticmethod
    def backward(self, gradOutput):
        x, vi, memory, y, params = self.saved_tensors
        batchSize = gradOutput.size(0)
        T = params[0].item()
        momentum = params[1].item()
        
        # add temperature
        gradOutput.data.div_(T)

        # gradient of linear
        gradInput = torch.mm(gradOutput.data, vi)
        gradInput.resize_as_(x)

        # update the non-parametric data
        weight_pos = memory.index_select(0, x_index.data.view(-1)).resize_as_(x)
        weight_pos.mul_(momentum)
        weight_pos.add_(torch.mul(x.data, 1-momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        memory.index_copy_(0, x_index, updated_weight)
        
        return gradInput, None, None, None

class LinearAverageContinued(nn.Module):

    def __init__(self, inputSize, outputSize,  lastMemory, inv_map, T=0.07, momentum=0.5):
        super(LinearAverageContinued, self).__init__()
        stdv = 1 / math.sqrt(inputSize)
        self.nLem = outputSize
        self.inv_map = inv_map
        self.register_buffer('params',torch.tensor([T, momentum]));
        self.register_buffer('memory', lastMemory)

    def forward(self, x, x_index, y):
        out = LinearAverageOpContinued.apply(x, x_index, y, self.memory, self.params, self.inv_map)
        return out