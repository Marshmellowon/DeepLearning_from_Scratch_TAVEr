import numpy as np
from NeuralNet2.TwoLayerNN import TwoLayer
from dataset.mnist import load_mnist
import matplotlib.pylab as plt

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []

# 하이퍼파라미터
iters_num = 1000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

network = TwoLayer(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    # 미니배치 구성
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    grad = network.numerical_gradient(x_batch, t_batch)

    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

print(train_loss_list)
y = train_loss_list
x = np.arange(0, 1000)
plt.plot(x, y)
plt.show()
