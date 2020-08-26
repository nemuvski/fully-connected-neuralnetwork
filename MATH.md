数式に対して、NumPyを利用して実装したものを記載する。

`np`とする。

```python
import numpy as np
```

---

内積

重みパラメータ$w$(行列)と入力ベクトル$x$の内積を求める場合を示す。

このとき、$w$のサイズは$m \times n$として、$x$のサイズは$n \times 1$とすると、$u$は$m \times 1$のベクトルになる。

```math
u = wx
```

```python
u = np.dot(w, x)
```

---

シグモイド関数

```math
f(u) = \frac{1}{1 + e^{-u}}
```

```python
def sigmoid(u):
  return 1 / (1 + np.exp(-u))
```

微分

```math
f'(u) = (1 - f(u)) f(u)
```

```python
def d_sigmoid(u):
  return (1 - sigmoid(u)) * sigmoid(u)
```

---

誤差関数

```math
E = \frac{1}{2} \sum_k (y_k - t_k)^2
```

```python
def error(y, t):
  return np.sum((y - p) ** 2) / 2
```

---

重みパラメータの更新式

$l$層目の重みを$w^{(l)}$と表す。
$L$層目の重みを$w^{(L)}$と表す。これは出力層である。

```math
w_{i,j}^{(l)} \leftarrow w_{i,j}^{(l)} - \rho \delta_{i}^{(l)} o_{j}^{(l-1)}
```

上記の式のうち、$\delta$の内容は層によって異なる。

```math
\delta_{i}^{(l)} = \left\{
\begin{array}{ll}
(o_{i}^{(l)} - t_{i}) f'(u_{i}^{(l)}) & (l = L) \\
\sum_{k} \delta_{k}^{(l+1)} w_{k,i}^{(l+1)} f'(u_{i}^{l}) & (l \lt L)
\end{array}
\right.
```

上記をPythonで書き表してみると... (ベクトル・行列の形状は、ここでは特に考慮せずに書く)

```python
# Lは出力層の番号を表す
def calc_delta(l, u, t, next_layer_w, next_layer_delta):
  if l == L:
    o = sigmoid(u)
    result = (o - t) * d_sigmoid(u)
  else:
    result = np.dot(next_layer_w, next_layer_delta) * d_sigmoid(u)
  return result

delta = calc_delta(l, u, t, next_layer_w, next_layer_delta)
w -= rho * np.dot(delta, o)
```
