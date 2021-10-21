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
        """
        return relu(Wx), summed along the axis.
        :param x: 2X1 vector
        :return:
        """
        return torch.sum(torch.relu(torch.matmul(self.W, x.float())))


def two_vec(x, y, func):
    """
    calculate h(x,y) on meshgrid
    """
    z = torch.zeros(SIZE, SIZE)
    for i in range(SIZE):
        for j in range(SIZE):
            z[i, j] = func(torch.tensor([x[i, j], y[i, j]]))
    return z


def plot_meshgrid(func):
    # set up points for evaluations
    x = np.linspace(-1, 1, SIZE)
    y = np.linspace(-1, 1, SIZE)
    X, Y = np.meshgrid(x, y)

    # evaluate points
    Z = (two_vec(torch.from_numpy(X), torch.from_numpy(Y), func))

    # set up visualisation
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, SIZE, cmap='binary')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$h(x_1,x_2)$')
    plt.show()


if __name__ == '__main__':
    func = h(torch.normal(torch.zeros(2, 2), torch.ones(2, 2)))
    plot_meshgrid(func)
