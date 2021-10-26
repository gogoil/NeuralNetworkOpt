import torch
import numpy as np


def get_theta(w, w_norm, v, v_norm):
    return torch.sum((w.T * v)) / (v_norm * w_norm)


def g(w, v):
    w_norm = torch.norm(w)
    v_norm = torch.norm(v)
    theta = get_theta(w, w_norm, v, v_norm)
    element_a = (v_norm / w_norm) * torch.sin(theta) * w
    element_b = theta * v
    return ((1 / (2 * np.pi)) * (element_a - element_b)) + (v / 2)


class GradientCalculator:
    def __init__(self, v, d):
        self.v = v
        self.v_norm = torch.norm(v)
        self.d = d

    def get_grad(self, w):
        element_a = (1 / 2) * w
        element_b = torch.zeros((self.d, self.d))
        element_c = torch.zeros((self.d, self.d))

        for i in range(self.d):
            for j in range(self.d):
                if j != i:
                    element_b[i] += g(w[i], w[j])
        for i in range(self.d):
            for j in range(self.d):
                element_c[i] += g(w[i], self.v[j])
        return element_a + element_b - element_c


def gradient_descent(w, num_of_steps, grad_calc, lr=1e-3):
    min_i = 0
    min_w = w
    min_grad_norm = 1e10
    with torch.no_grad():
        cur_w = w
        for i in range(num_of_steps):
            prev_w = cur_w
            grad = grad_calc.get_grad(prev_w)
            cur_w += -lr * grad
    #         grad_norm = torch.norm(grad)
    #         print(f"the grad is\n{grad}\nin iteration {i} and the norm is "
    #               f"\n{grad_norm}\n")
    #         if grad_norm < min_grad_norm:
    #             min_grad_norm = grad_norm
    #             min_i = i
    #             min_w = cur_w
    # print(f"min value: i{min_i}, min_grad_norm{min_grad_norm} and min w:\n"
    #       f"{min_w}" )
    return cur_w
