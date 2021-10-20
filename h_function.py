import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

SIZE = 50

class h(nn.Module):
    """
    :arg W 2 on 2 matrix
    """
    def __init__(self, W):
        super(h, self).__init__()
        self.W = W

    def forward(self, x):
        return torch.sum(torch.relu(torch.matmul(self.W, x.float())))

func = h(torch.normal(torch.zeros(2,2),torch.ones(2,2)))

def two_vec(x,y):
    z = torch.zeros(SIZE, SIZE)
    for i in range(SIZE):
        for j in range(SIZE):
            z[i,j] = func(torch.tensor([x[i,j],y[i,j]]))
    return z

def plot_meshgrid():
    x = np.linspace(-1, 1, SIZE)
    y = np.linspace(-1, 1, SIZE)

    X, Y = np.meshgrid(x, y)
    Z = (two_vec(torch.from_numpy(X),torch.from_numpy(Y)))
    ax = plt.axes(projection='3d')

    ax.contour3D(X, Y, Z, 50 , cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

if __name__ == '__main__':
    plot_meshgrid()