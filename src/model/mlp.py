import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, act, net_dim, input_dim=784, out_dim=10):
        super(MLP, self).__init__()
        self.act = act
        self.net_dim = net_dim
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.net = self.make_net(act, net_dim, out_dim)
    
    def make_net(self, act, net_dim, out_dim):
        if act == "Hardswish":
            act = nn.Hardswish
        elif act == "ReLU":
            act = nn.ReLU
        net = []
        last_dim = self.input_dim
        for i in range(len(net_dim)):
            if i == len(net_dim) - 1:
                net += [nn.Linear(last_dim, out_dim), act()]
            else:
                net += [nn.Linear(last_dim, net_dim[i]), act()]
            last_dim = net_dim[i]
        return nn.Sequential(*net)

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        return self.net(x)

if __name__ == "__main__":
    act = "Hardswish"
    net_dim = [256, 64, 32]
    mlp = MLP(act, net_dim)
    print(mlp)
    x = torch.randn(2, 784)
    print(mlp(x).shape)