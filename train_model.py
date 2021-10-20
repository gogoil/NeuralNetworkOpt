import torch
import model

d = 10

NUMBER_OF_ITERATIONS = 1000


def loss_f(output, target):
    loss = torch.sum((output - target) ** 2)
    return loss


def generate_y(teacher_net, X):
    return teacher_net(X)


def train(model, loss_fn, optimizer, point_sampler, traget, teacher_net):
    model.train()
    for i in range(NUMBER_OF_ITERATIONS):
        X = point_sampler()
        # Compute prediction error
        pred = model(X)
        y = traget(teacher_net, X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.item()
        print(f"loss: {loss:>7f} in iteration {i}")


def point_sampler():
    return torch.normal(torch.zeros(d), torch.ones(d))


if __name__ == '__main__':
    student_net = model.TwoLayerNeuralNetwork(d)

    optimizer = torch.optim.SGD(student_net.parameters(), lr=1e-3)

    teacher_net = model.generate_teacher_model_normal(d)
    train(student_net, loss_f, optimizer, point_sampler, generate_y,
          teacher_net)
