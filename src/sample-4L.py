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

    seed_value = 5839
    random_generator = np.random.RandomState(seed_value)
    X_dim = 4
    v_min = 0.0
    v_max = 0.5
    threshold_value = 1.5

    # 学習データの作成
    X_train = utils.create_threshold_X((500, X_dim), v_min, v_max,
                                       random_generator)
    Y_train = utils.create_threshold_Y(X_train, threshold_value)

    # 評価データの作成
    X_test = utils.create_threshold_X((100, X_dim), v_min, v_max,
                                      random_generator)
    Y_test = utils.create_threshold_Y(X_test, threshold_value)

    # モデルについて
    learning_rate = 0.1
    epoch = 1000
    num_hidden_units = [4, 3]
    layer_model = [X_train.shape[1],
                   num_hidden_units[0],
                   num_hidden_units[1],
                   Y_train.shape[1]]
    model = NeuralNetwork(layer_model, seed_value=seed_value)
    errors = model.train(learning_rate, epoch, X_train, Y_train, verbose=True)
    predictions = model.predict(X_test)

    utils.print_accuracy_rate(Y_test, predictions)
    utils.plot_error_log(errors, epoch, 'chart/error_chart_sample-4L.png')


if __name__ == '__main__':
    main()
