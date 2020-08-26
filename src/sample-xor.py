# -*- coding: utf-8 -*-

""""
ニューラルネットワーク・サンプル
"""

import os
import sys
import numpy as np


def main():
    sys.path.append(os.path.join(os.getcwd(), 'src', 'lib'))
    from neuralnetwork import NeuralNetwork
    import utils

    seed_value = 8976

    # 学習データの作成 (評価にも利用)
    X_train = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=np.float32)
    Y_train = np.array([
        [1, 0],
        [0, 1],
        [0, 1],
        [1, 0]
    ], dtype=np.float32)

    # モデルについて
    learning_rate = 0.4
    epoch = 2000
    num_hidden_units = 4
    layer_model = [X_train.shape[1],
                   num_hidden_units,
                   Y_train.shape[1]]
    model = NeuralNetwork(layer_model, seed_value=seed_value)
    errors = model.train(learning_rate, epoch, X_train, Y_train, verbose=True)
    predictions = model.predict(X_train)

    utils.print_accuracy_rate(Y_train, predictions)
    utils.plot_error_log(errors, epoch, 'chart/error_chart_sample-xor.png')


if __name__ == '__main__':
    main()
