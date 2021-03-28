import numpy as np
import matplotlib.pyplot as plt


def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] - pt[1]) / 1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)


def generate_XOR_easy():
    inputs = []
    labels = []
    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue

        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21, 1)


def show_result(x, y, pred_y):
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.show()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def derivative_sigmoid(x):
    return np.multiply(x, 1.0 - x)


class Layer():
    def __init__(self, input_size, output_size):
        self.w = np.random.normal(0, 1, (input_size+1, output_size))

    def forward(self, x):
        x = np.append(x, np.ones((x.shape[0], 1)), axis=1)
        self.forward_gradient = x
        self.y = sigmoid(x @ self.w)
        return self.y

    def backward(self, derivative_C):
        self.backward_gradient = derivative_sigmoid(self.y) @ derivative_C
        return self.backward_gradient @ self.w[:-1].T

    def update(self, learning_rate):
        self.gradient = self.forward_gradient.T @ self.backward_gradient
        self.w -= learning_rate * self.gradient
        return self.gradient


class NN():
    def __init__(self, sizes, learning_rate=0.1):
        self.learning_rate = learning_rate
        sizes2 = sizes[1:] + [0]
        self.l = []
        for a, b in zip(sizes, sizes2):
            if (a+1)*b == 0:
                continue
            self.l += [Layer(a, b)]

    def forward(self, x):
        _x = x
        for l in self.l:
            _x = l.forward(_x)
        return _x

    def backward(self, dC):
        _dC = dC
        for l in self.l[::-1]:
            _dC = l.backward(_dC)

    def update(self):
        gradients = []
        for l in self.l:
            gradients += [l.update(self.learning_rate)]
        return gradients


if __name__ == '__main__':
    x, y = generate_linear(n=100)  # (100, 2), (100, 1)
    x = np.array(x)
    y = np.array(y)
    print(x.shape)
    print(y.shape)
