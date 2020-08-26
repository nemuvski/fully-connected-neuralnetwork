# -*- coding: utf-8 -*-

""""
サンプルコードのためのユーティリティ
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_error_log(error_log, epoch, filename):
    """誤差値の変動をグラフに保存

    Parameters
    ----------
    error_log: 誤差値の配列
    epoch: 学習回数
    filename: 画像名 (PNG)

    Returns
    -------
    なし
    """
    # 横軸を回数, 縦軸を誤差
    x = np.arange(1, epoch + 1)
    y = np.array(error_log)
    plt.plot(x, y)
    plt.xlabel('Epoch')
    plt.ylabel('Square Error')
    # plt.show()
    plt.savefig(filename)


def print_accuracy_rate(Y, P):
    """評価データでの正解率[%]を算出して印字

    Parameters
    ----------
    Y: 正解ベクトル
    P: 予測ベクトル

    Returns
    -------
    なし
    """
    num_data = Y.shape[0]
    num_correct = 0
    for y, p in zip(Y, P):
        num_correct += int(np.argmax(y) == np.argmax(p))
    accuracy_rate = num_correct / num_data * 100
    print('Accuracy Rate: {0}[%]'.format(accuracy_rate))


def create_threshold_X(shape, min, max, random_gen):
    """特徴ベクトルを作成 (乱数)

    Parameters
    ----------
    shape: (サンプル数, 次元数)
    min: 最小値
    max: 最大値
    random_gen: 乱数生成器

    Returns
    -------
    ベクトル群 (ndarray)
    """
    return (max-min) * random_gen.random_sample(size=shape) + min


def create_threshold_Y(X, threshold_value):
    """正解ベクトルの生成 (特徴ベクトルの数値の合計が閾値以上の場合は1とする)

    Parameters
    ----------
    X: 特徴ベクトル群
    threshold_value: 閾値

    Returns
    -------
    ベクトル群 (ndarray)
    """
    label_Y = np.sum(X, axis=1) >= threshold_value
    Y = []
    for label_y in label_Y:
        y = [0, 0]
        y[int(label_y)] = 1
        Y.append(y)
    return np.array(Y, dtype=np.float32)
