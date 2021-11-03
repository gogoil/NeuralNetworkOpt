import torch
import numpy as np
import model
from scipy.optimize import approx_fprime

one_over_2_pi = 1 / (2 * np.pi)


def get_theta(w, w_norm, v, v_norm):
    angle = (w.T @ v) / (v_norm * w_norm)
    angle = np.clip(angle, -1, 1)
    return np.arccos(angle)


def g(w, v):
    """subroutine for calc gradient"""
    w_norm = torch.linalg.norm(w)
    v_norm = torch.linalg.norm(v)
    theta = get_theta(w, w_norm, v, v_norm)
    element_a = (v_norm / w_norm) * np.sin(theta) * w
    element_b = (np.pi - theta) * v
    return one_over_2_pi * (element_a + element_b)


def f(w, v):
    """subroutine for calc target function"""
    w_norm = torch.linalg.norm(w)
    v_norm = torch.linalg.norm(v)
    theta = get_theta(w, w_norm, v, v_norm)
    element_a = one_over_2_pi * w_norm * v_norm
    element_b = np.sin(theta)
    element_c = (np.pi - theta) * np.cos(theta)
    return element_a * (element_b + element_c)


class GradientCalculator:
    def __init__(self, v, d):
        self.v = v
        self.v_norm = torch.linalg.norm(v)
        self.d = d

    def target_function(self, w):
        element_a = 0
        element_b = 0
        element_c = 0
        for i in range(self.d):
            for j in range(self.d):
                element_a += f(w[i], w[j])
                element_b += f(w[i], self.v[j])
                element_c += f(self.v[i], self.v[j])
        return (0.5 * element_a) + (-element_b) + (0.5 * element_c)

    def get_grad(self, w):
        """
        return the gradient of matrix w
        """
        element_a = 0.5 * w
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


def gradient_descent(x0, num_of_steps, grad_calc, d, lr=1e-2):
    """gradient descent implementation, grad_calc have method get_grad
    that gets a point and returns its gradient"""
    path_list = torch.zeros((num_of_steps, d, d))
    with torch.no_grad():
        cur_x0 = x0
        for i in range(num_of_steps):
            prev_x0 = cur_x0
            grad = grad_calc.get_grad(prev_x0)
            cur_x0 += -(lr * grad)
            path_list[i] = cur_x0
    return path_list


if __name__ == '__main__':
    with torch.no_grad():
        student = model.generate_teacher_model_Id(10).linearLayer.weight
        teacher = model.generate_teacher_model_normal(
            10).linearLayer.weight

        # check angle between vectors is calculated properly

        # print(f"w matrix\n{w}")
        # print(f"v matrix\n{v}")
        # print("mine, ", get_theta(w, torch.linalg.norm(w), v, torch.linalg.norm(v)))
        # print("numpy ", angle_between(w.detach().numpy(), v.detach().numpy()))

        # check target function is functioning properly
        GC = GradientCalculator(teacher, 10)
        print(GC.get_grad(teacher))
