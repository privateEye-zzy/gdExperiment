# 梯度下降法（沿着目标函数梯度下降的方向搜索极小值）
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['FangSong']
plt.rcParams['axes.unicode_minus'] = False
# 目标函数 x^2 + 2 * x + 10
def func(x):
    return np.square(x) + 2 * x + 10
# 目标函数的一阶导数
def dfunc(x):
    return 2 * x + 2
# 学习率因子-梯度下降
def gd_learnrate(x_start, dfunc, epochs, learnrate):
    y = np.zeros(epochs + 1)
    y[0] = x_start
    for i in range(epochs):
        y[i+1] = y[i] - learnrate * dfunc(y[i])
    return y
# 冲量因子-梯度下降
def gd_momentum(x_start, dfunc, epochs, learnrate, momentum):
    y = np.zeros(epochs + 1)
    y[0] = x_start
    v = 0
    for i in range(epochs):
        p = v * momentum  # 速度变化过程产生的冲量
        dv = -learnrate * dfunc(y[i])  # 本次的梯度下降量
        v = dv + p  # 本次的速度增量，受到冲量p的影响，做加速或者减速运动
        y[i + 1] = y[i] + v
    return y
# 学习率衰减因子-梯度下降
def gd_decay(x_start, dfunc, epochs, learnrate, decay):
    y = np.zeros(epochs + 1)
    y[0] = x_start
    for i in range(epochs):
        learnrate_decay = learnrate * 1.0 / (1.0 + decay * i)
        y[i + 1] = y[i] - learnrate_decay * dfunc(y[i])
    return y
# 测试学习率因子实验
def do_gd_learnrate():
    line_x = np.linspace(-5, 5, 100)
    line_y = func(line_x)
    x_start, epochs = -5, 10
    learnRate = [0.1, 0.3, 0.9]
    color = ['r', 'g', 'y']
    for i in range(len(learnRate)):
        x = gd_learnrate(x_start, dfunc, epochs, learnrate=learnRate[i])
        plt.subplot(1, 3, i + 1)
        plt.plot(line_x, line_y, c='b')
        plt.plot(x, func(x), c=color[i], label='lr={}'.format(learnRate[i]))
        plt.scatter(x, func(x), c=color[i])
        plt.legend()
    plt.show()
# 测试冲量因子实验
def do_gd_momentum():
    line_x = np.linspace(-5, 5, 100)
    line_y = func(line_x)
    x_start, epochs = -5, 10
    learnRate = [0.01, 0.1, 0.6, 0.9]
    momentum = [0.0, 0.1, 0.5, 0.9]
    color = ['k', 'r', 'g', 'y']
    row, col = len(learnRate), len(momentum)
    for i in range(row):
        for j in range(col):
            x = gd_momentum(x_start, dfunc, epochs, learnrate=learnRate[i], momentum=momentum[j])
            plt.subplot(row, col, i * col + j + 1)
            plt.plot(x, func(x), c=color[i], label='lr={}, mo={}'.format(learnRate[i], momentum[j]))
            plt.plot(line_x, line_y, c='b')
            plt.scatter(x, func(x), c=color[i])
            plt.legend()
    plt.show()
# 测试学习率衰减因子实验
def do_gd_decay():
    line_x = np.linspace(-5, 5, 100)
    line_y = func(line_x)
    x_start, epochs = -5, 10
    learnRate = [0.1, 0.3, 0.9, 0.99]
    decay = [0.0, 0.01, 0.5, 0.9]
    color = ['k', 'r', 'g', 'y']
    row, col = len(learnRate), len(decay)
    for i in range(row):
        for j in range(col):
            x = gd_decay(x_start, dfunc, epochs, learnrate=learnRate[i], decay=decay[j])
            plt.subplot(row, col, i * col + j + 1)
            plt.plot(x, func(x), c=color[i], label='lr={}, de={}'.format(learnRate[i], decay[j]))
            plt.plot(line_x, line_y, c='b')
            plt.scatter(x, func(x), c=color[i])
            plt.legend()
    plt.show()
if __name__ == "__main__":
    # do_gd_learnrate()
    # do_gd_momentum()
    do_gd_decay()
