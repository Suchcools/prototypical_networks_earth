import torch.nn as nn
import torch

def conv_block(in_channels, out_channels):
    '''
    returns a block conv-bn-relu-pool
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels,(2,2), padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class ProtoNet(nn.Module):

    def __init__(self, x_dim=1, hid_dim=64, z_dim=64):
        super(ProtoNet, self).__init__()
        self.conv1=conv_block(x_dim, hid_dim)
        self.conv2=conv_block(hid_dim, hid_dim)
        self.conv3=conv_block(hid_dim, z_dim)
        self.output = nn.Linear(4032,z_dim)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        return self.output(x)
# model = ProtoNet() # 40 1 28 28 -> 40 64
# print(model(torch.Tensor(1,1,6,500)).shape)