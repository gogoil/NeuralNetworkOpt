import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda


class TwoLayerNeuralNetwork(nn.Module):
    def __init__(self, d):
        super(TwoLayerNeuralNetwork, self).__init__()
        self.linearLayer = nn.Linear(d, d, bias=False)
        self.two_layer_relu = nn.Sequential(
            self.linearLayer,
            nn.ReLU(),
            # nn.Flatten()
        )

    def forward(self, x):
        return self.two_layer_relu(x)


def generate_teacher_model_Id(d):
    teacher_net = TwoLayerNeuralNetwork(d)
    teacher_net.linearLayer.weight = torch.nn.Parameter(torch.eye(d))
    return teacher_net


def generate_teacher_model_normal(d):
    teacher_net = TwoLayerNeuralNetwork(d)
    teacher_net.linearLayer.weight = torch.nn.Parameter(torch.normal(
        torch.zeros((d, d)), torch.ones((d, d))))
    return teacher_net


def generate_teacher_model_alpha_plus_beta(d, a, b):
    teacher_net = TwoLayerNeuralNetwork(d)
    v = (torch.zeros((d,d))+a) + (torch.eye(d)*b)
    teacher_net.linearLayer.weight = torch.nn.Parameter(v)
    return teacher_net

