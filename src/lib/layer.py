# -*- coding: utf-8 -*-

""""
ニューラルネットワークの層用のモジュール
"""

import sys
import numpy as np


class Layer:
    """層の基底クラス"""

    def __init__(self, input_dim, num_units, random_generator=None):
        """重みパラメータの初期化

        Parameters
        ----------
        input_dim: 入力するベクトルの次元数
        num_units: ユニット数
        random_generator=None: 乱数生成器

        Returns
        -------
        なし
        """
        if input_dim < 1 or num_units < 1:
            print('入力するベクトルの次元数, ユニット数は1以上を指定してください')
            sys.exit(1)

        # 入力されたベクトル (前層の出力ベクトルとも言える)
        self.input_param = None
        # 出力ベクトル
        self.output_param = None

        # 重みパラメータ行列のサイズ
        size = (num_units, input_dim + 1)
        # 乱数生成器の選択
        r = random_generator if random_generator is not None else np.random
        # 重みパラメータ行列の生成
        self.w = r.standard_normal(size=size)

    def forward(self, x):
        """重みと入力ベクトルの内積値をシグモイド関数へ入力して、その出力値を返却

        Parameters
        ----------
        x: 入力ベクトル (バイアス項の分は内部で追加される)

        Returns
        -------
        シグモイド関数の出力ベクトル
        """
        # 入力ベクトルと出力ベクトルは内部で保持しておく
        self.input_param = np.c_[x, np.ones((1, 1))]
        self.output_param = self.sigmoid(np.dot(self.input_param,
                                                self.w.T))
        return self.output_param

    @staticmethod
    def sigmoid(x):
        """シグモイド関数

        Parameters
        ----------
        x: ベクトルでも可

        Returns
        -------
        計算結果
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def d_sigmoid(x):
        """シグモイド関数の微分式

        Parameters
        ----------
        x: ベクトルでも可

        Returns
        -------
        計算結果
        """
        return (1 - x) * x


class HiddenLayer(Layer):
    """隠れ層クラス"""

    def back(self, learning_rate, following_layer_w, following_layer_delta):
        """"重みパラメーターの更新し、その際に算出したデルタの内容を返却

        Parameters
        ----------
        learning_rate: 学習率
        following_layer_w: 1つ先の層の重みパラメータ
        following_layer_delta: 1つ先の層のデルタの内容

        Returns
        -------
        重みパラメータ, デルタ関数の結果 (前層の重みパラメータの更新で利用するため)
        """
        delta = self.__calc_delta(following_layer_w, following_layer_delta)
        self.w -= learning_rate * np.dot(delta.T, self.input_param)
        return (self.w, delta)

    def __calc_delta(self, following_layer_w, following_layer_delta):
        """デルタ関数

        Parameters
        ----------
        following_layer_w: 1つ次の層の重みパラメータ
        following_layer_delta: 1つ次の層のデルタの内容

        Returns
        -------
        計算結果
        """
        temp = self.d_sigmoid(self.output_param)
        return np.dot(following_layer_delta, following_layer_w[:, :-1]) * temp


class OutputLayer(Layer):
    """出力層クラス"""

    def back(self, learning_rate, y):
        """重みパラメーターの更新し、その際に算出したデルタの内容を返却

        Parameters
        ----------
        learning_rate: 学習率
        y: 教師ベクトル

        Returns
        -------
        重みパラメータ, デルタ関数の結果 (前層の重みパラメータの更新で利用するため)
        """
        # デルタの算出
        delta = self.__calc_delta(y)
        # 重みパラメーターの更新
        self.w -= learning_rate * np.dot(self.input_param.T, delta).T
        return (self.w, delta)

    def __calc_delta(self, y):
        """デルタ関数

        Parameters
        ----------
        y: 教師ベクトル

        Returns
        -------
        計算結果
        """
        temp = self.d_sigmoid(self.output_param)
        return (self.output_param - y) * temp
