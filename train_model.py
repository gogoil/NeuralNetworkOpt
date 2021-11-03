import matplotlib.pyplot as plt
import torch
import model
import GDoptimizer
import numpy as np

d = 10
lr = 1e-2
NUMBER_OF_ITERATIONS = 1000


def normalize(mat):
    return mat / torch.linalg.norm(mat)


def loss_f(output, target):
    """squared loss"""
    loss = torch.sum((output - target) ** 2)
    return loss


def train(model, loss_fn, optimizer, point_sampler, teacher_net):
    model.train()
    for i in range(NUMBER_OF_ITERATIONS):
        X = point_sampler()
        # Compute prediction error
        pred = model(X)
        y = teacher_net(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.item()
        print(f"loss: {loss:>7f} in iteration {i}")


def plot_lost(w, v):
    loss = np.zeros(NUMBER_OF_ITERATIONS)
    for i in range(NUMBER_OF_ITERATIONS):
        loss[i] = loss_f(w[i], v)
    plt.plot(loss)
    plt.ylabel("loss")
    plt.xlabel("iteration number")
    plt.show()


def point_sampler():
    """
    :return: sampled data
    """
    return torch.normal(torch.zeros(d), torch.ones(d))


if __name__ == '__main__':
    student_net = model.TwoLayerNeuralNetwork(d)
    teacher_net = model.generate_teacher_model_normal(d)

    opt = GDoptimizer.GradientCalculator(teacher_net.linearLayer.weight, d)
    path_list = GDoptimizer.gradient_descent(student_net.linearLayer.weight,
                                            d=d,
                                     grad_calc=opt,
                                     num_of_steps=NUMBER_OF_ITERATIONS,
                                     lr=lr)
    plot_lost(path_list, teacher_net.linearLayer.weight)
    w = path_list[NUMBER_OF_ITERATIONS - 1, :, :]
    loss = loss_f(w, teacher_net.linearLayer.weight)
    print(f"loss {loss}")
    print(f"w matrix\n{w}")
    print(f"w matrix normalized\n{normalize(w)}")

    with open("matrix_minima.txt", "a") as f:
        f.write(f"loss {loss}\n")
        f.write(f"w matrix\n{w}\n")
        f.write(f"w matrix normalized\n{normalize(w)}\n")
        f.close()
