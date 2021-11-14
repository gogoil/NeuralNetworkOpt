import matplotlib.pyplot as plt
import torch
from scipy.misc import derivative

import calc_f_g_numpy_only as np_calc
import model
import GDoptimizer
import numpy as np

d = 10


def test_gradient_of_subroutine_f_and_g_numpy_version():
    w = np.random.normal(0, 1, d)
    assert_der_similar(
        f=lambda x: [np_calc.f(x)],
        df=lambda x: [np_calc.g(x)],
        x0=w)


def test_gradient_of_target_func_numpy_version():
    v = np.eye(d)
    grad_calc = np_calc.GradientCalculator(v, d)
    w = np.random.normal(0, 1, (d, d))
    assert_der_similar(
        f=lambda x: [grad_calc.target_function(x.reshape(d, d))],
        df=lambda x: [grad_calc.gradient(x.reshape(d, d))],
        x0=w.reshape(d ** 2))


def test_gradient_of_subroutine_f_and_g():
    w = torch.normal(torch.zeros((d, d)), torch.ones((d, d)))
    with torch.no_grad():
        assert_der_similar(
            f=lambda x: [GDoptimizer.build_f_subroutine(x.reshape((d, d)))],
            df=lambda x: [GDoptimizer.build_g_subroutine(x.reshape((d, d)))],
            x0=w.flatten())


def test_gradient_of_target_func():
    student_net = model.TwoLayerNeuralNetwork(d)

    teacher_net = model.generate_teacher_model_Id(d)

    opt = GDoptimizer.GradientCalculator(teacher_net.linearLayer.weight, d)

    print()
    with torch.no_grad():
        assert_der_similar(
            f=lambda x: [opt.target_function(x.reshape((d, d)))],
            df=lambda x: [opt.get_grad(x.reshape((d, d)))],
            x0=student_net.linearLayer.weight.flatten())


def assert_der_similar(f, df, x0, tol=.001):
    x = differential(f, x0)
    y = df(x0)
    y = y[0].reshape(d ** 2)  # cpu().detach().numpy()
    error = np.max(np.abs(x - y))

    if error <= tol:
        print("[SUCCESS] error:{0}".format(error))
        return

    print("[FAILURE] error:{0}".format(error))
    print(f"my gradient {y.reshape(d, d)}\n")
    print(f"required gradient {x.reshape(d, d)}\n")

    print(f"normalized {y.reshape(d, d) / x.reshape(d, d)}")
    assert False


def differential(f, x0):
    n_vars = len(x0)
    n_funcs = len(f(x0))

    der = np.zeros((n_funcs, n_vars))
    for i_var in range(n_vars):
        for i_func in range(n_funcs):
            der[i_func, i_var] = partial_derivative(f, i_func, i_var, x0)

    return der


def partial_derivative(func, i_func, i_var, point):
    args = point[:]

    def _func(x):
        args[i_var] = x
        return func(args)[i_func]

    return derivative(_func, point[i_var], dx=1e-6)

# def _target_small():
#     np.random.seed(0)
#     b = SeungNelsonBuilder(3)
#     m = b.t.to_manifold()
#     m.vertices += np.random.ranf((m.num_vertices, 3)) * 0.1
#     return m
