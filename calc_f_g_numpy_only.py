import numpy as np

one_over_2pi = 1 / (2 * np.pi)
d = 10


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'"""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def f(w, v=np.ones(d)):
    """subroutine of the target function"""
    w_norm = np.linalg.norm(w)
    v_norm = np.linalg.norm(v)
    arg = angle_between(w, v)
    ans = one_over_2pi * w_norm * v_norm * (
            np.sin(arg) + (np.cos(arg) * (np.pi - arg)))
    return ans


def g(w, v=np.ones(d)):
    """subroutine of the gradient function"""
    w_norm = np.linalg.norm(w)
    v_norm = np.linalg.norm(v)
    arg = angle_between(w, v)
    ans = one_over_2pi * ((v_norm * np.sin(arg) * (w / w_norm)) + (v * (np.pi
                                                                        - arg)))
    return ans


class GradientCalculator:
    def __init__(self, v, d):
        self.v = v
        self.v_norm = np.linalg.norm(v)
        self.d = d

    def target_function(self, w):
        element_w = 0
        element_w_v = 0
        element_v = 0
        for i in range(self.d):
            for j in range(self.d):
                element_w += f(w[i], w[j])
                element_w_v += f(w[i], self.v[j])
                element_v += f(self.v[i], self.v[j])
        return (0.5 * element_w) + (- element_w_v) + (0.5 * element_v)

    def gradient(self, w):
        element_w = np.zeros((self.d, self.d))
        element_w_v = np.zeros((self.d, self.d))
        for i in range(self.d):
            for j in range(self.d):
                element_w_v[i] += g(w[i], self.v[j])
                if i != j:
                    element_w[i] += g(w[i], w[j])

        return (0.5 * w) + element_w - element_w_v


def gradient_descent(x0, num_of_steps, grad_calc, d, lr=1e-2):
    """gradient descent implementation, grad_calc have method get_grad
    that gets a point and returns its gradient"""
    path_list = np.zeros((num_of_steps, d, d))
    gradient_norm_list = np.zeros(num_of_steps)
    cur_x0 = x0
    for i in range(num_of_steps):
        prev_x0 = cur_x0
        grad = grad_calc.gradient(prev_x0)
        # print(f"{i} grad norm is {np.linalg.norm(grad)}")
        cur_x0 += -(lr * grad)
        path_list[i] = cur_x0
        gradient_norm_list[i] = np.linalg.norm(grad)
    return path_list , gradient_norm_list
