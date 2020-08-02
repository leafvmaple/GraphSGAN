import torch
from torch.nn.parameter import Parameter
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from utils.functional import LinearWeightNorm
class Discriminator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.layers = torch.nn.ModuleList([
            LinearWeightNorm(input_dim, 500),
            LinearWeightNorm(500, 500),
            LinearWeightNorm(500, 250),
            LinearWeightNorm(250, 250),
            LinearWeightNorm(250, 250)]
        )
        self.final = LinearWeightNorm(250, output_dim, weight_scale=1)
    def forward(self, x, feature = False):
        x = x.view(-1, self.input_dim)
        noise = torch.randn(x.size()) * 0.05 if self.training else torch.Tensor([0]).cuda()
        x = x + Variable(noise, requires_grad = False)
        for i in range(len(self.layers)):
            m = self.layers[i]
            x_f = F.elu(m(x))
            noise = torch.randn(x_f.size()) * 0.5 if self.training else torch.Tensor([0]).cuda()
            x = (x_f + Variable(noise, requires_grad = False))
        if feature:
            return self.final(x), x_f
        return self.final(x)