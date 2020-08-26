# -*- coding: utf-8 -*-

""""
ニューラルネットワーク用のモジュール
"""

import sys
import numpy as np
from layer import HiddenLayer, OutputLayer


class NeuralNetwork:
    """ニューラルネットワーククラス"""

    def __init__(self, layer_model, seed_value=None):
        """モデルの構築, 各層のオブジェクトの生成

        Parameters
        ----------
        layer_model: リスト (各層のユニット数を入れる)
        seed_value=None: シード値

        Returns
        -------
        なし
        """
        if len(layer_model) < 3:
            print('3層以上を設定してください')
            sys.exit(1)

        # 乱数生成器の定義
        random_generator = None
        if seed_value is not None:
            random_generator = np.random.RandomState(seed_value)

        # 各層のオブジェクトを生成
        self.__layers = self.__construction_layers(layer_model,
                                                   random_generator)

    def train(self, learning_rate, epoch, X, Y, verbose=False):
        """学習

        Parameters
        ----------
        learning_rate: 学習率
        epoch: 学習回数
        X: 入力(特徴)ベクトル群
        Y: 正解ベクトル
        verbose=False: 途中経過を印字するか

        Returns
        -------
        誤差 (1エポック単位)
        """
        if X.shape[0] != Y.shape[0]:
            print('データの数が合っていません')
            sys.exit(1)

        num_data = X.shape[0]
        mean_errors = []

        for i in range(epoch):
            mean_error = 0
            for x, y in zip(X, Y):
                mean_error += self.__back_network(learning_rate, x, y)
            mean_error /= num_data
            mean_errors.append(mean_error)

            if verbose:
                print('{0}/{1}: {2}'.format(i+1, epoch, mean_error))

        return np.array(mean_errors)

    def predict(self, X):
        """ネットワークに入力して予測する

        Parameters
        ----------
        X: 予測する入力(特徴)ベクトル群

        Returns
        -------
        予測ベクトル群
        """
        P = []
        for x in X:
            P.append(self.__forward_network(x))
        return np.array(P)

    def __forward_network(self, x):
        """順方向 (各層の入力, 出力ベクトルが更新される)

        Parameters
        ----------
        x: 入力ベクトル

        Returns
        -------
        予測ベクトル
        """
        p = x.reshape((1, -1))  # 計算のために(1, x)の形に整形
        for l in self.__layers:
            p = l.forward(p)
        return p

    def __back_network(self, learning_rate, x, y):
        """重み更新

        Parameters
        ----------
        learning_rate: 学習率
        x: 入力ベクトル
        y: 正解ラベル

        Returns
        -------
        誤差
        """
        # はじめに各層の入力, 出力ベクトルを更新
        p = self.__forward_network(x)
        # 誤差計算
        error = self.__square_error(y, p)
        # 重みの更新
        w, delta = None, None
        for l in reversed(self.__layers):
            if w is None:
                # 出力層
                w, delta = l.back(learning_rate, y)
            else:
                # 中間層
                w, delta = l.back(learning_rate, w, delta)

        return error

    @staticmethod
    def __square_error(y, p):
        """二乗誤差 (※二乗和を2で割ったものを用いる)

        Parameters
        ----------
        y: 教師ベクトル
        p: 予測ベクトル

        Returns
        -------
        計算結果
        """
        return ((y - p) ** 2).sum(axis=1) / 2

    @staticmethod
    def __construction_layers(layer_model, random_generator):
        layers = []
        num_layers = len(layer_model)
        layer_object = None
        for i in range(1, num_layers):
            if i == num_layers - 1:
                layer_object = OutputLayer(layer_model[i - 1], layer_model[i],
                                           random_generator)
            else:
                layer_object = HiddenLayer(layer_model[i - 1], layer_model[i],
                                           random_generator)
            layers.append(layer_object)
        return layers
