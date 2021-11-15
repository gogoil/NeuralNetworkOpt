import matplotlib.pyplot as plt
import torch

import numpy as np
import calc_f_g_numpy_only as np_calc

d = 10
lr = 1e-2
NUMBER_OF_ITERATIONS = 1000


def normalize(mat):
    return mat / torch.linalg.norm(mat)


def loss_f(output, target):
    """squared loss"""
    return np.sum((output - target) ** 2)


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


def plot_lost(w: np.ndarray, v: np.ndarray):
    loss = np.zeros(NUMBER_OF_ITERATIONS)
    for i in range(NUMBER_OF_ITERATIONS):
        loss[i] = loss_f(w[i], v)
    plt.plot(loss)
    plt.ylabel("loss")
    plt.xlabel("iteration number")
    plt.show()


def plot_target_function(w: np.ndarray, grad_calc: np_calc.GradientCalculator):
    f_x = np.zeros(NUMBER_OF_ITERATIONS)
    for i in range(NUMBER_OF_ITERATIONS):
        f_x[i] = grad_calc.target_function(w[i])
    plt.plot(f_x)
    plt.ylabel("target function")
    plt.xlabel("iteration number")
    plt.show()
    for i in range(NUMBER_OF_ITERATIONS):
        if i % 100 == 0:
            print(f"i is {i} and loss{f_x[i]}")
    print(f"last {f_x[-1]}")


def plot_grad_norm(grad_norm_list: np.ndarray):
    plt.plot(grad_norm_list)
    plt.ylabel("gradient norm")
    plt.xlabel("iteration number")
    plt.show()


def point_sampler():
    """
    :return: sampled data
    """
    return torch.normal(torch.zeros(d), torch.ones(d))


def main():
    student_net = np.random.normal(0, 1, (d, d))

    teacher_net = np.eye(d)

    opt = np_calc.GradientCalculator(teacher_net, d)  # init gradient
    # calculator object
    path_list, gradient_norm_list = np_calc.gradient_descent(student_net,
                                                             d=d,
                                                             grad_calc=opt,
                                                             num_of_steps=NUMBER_OF_ITERATIONS,
                                                             lr=lr)
    # plot_lost(path_list, teacher_net)
    # plot_grad_norm(gradient_norm_list)
    plot_target_function(path_list, opt)
    # w = path_list[NUMBER_OF_ITERATIONS - 1, :, :]
    # loss = loss_f(w, teacher_net)
    # print(f"loss {loss}")
    # print(f"w matrix\n{w}")
    # print(f"w matrix normalized\n{normalize(w)}")

    # with open("matrix_minima.txt", "a") as f:
    #     f.write(f"loss {loss}\n")
    #     f.write(f"w matrix\n{w}\n")
    #     # f.write(f"w matrix normalized\n{normalize(w)}\n")
    #     f.close()


if __name__ == '__main__':
    main()
