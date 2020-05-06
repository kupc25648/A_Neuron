'''
========================
A BASIC NEURON
========================
'''
'''
------------------------
ENGLISH
------------------------
DIAGRAM
x -> [sigma(wx+b)] -> y_pred

SYMBOLS
[] : Neuron
x : Input of the neuron
sigma : activation funtion, using sigmoid function sigma(x) = (1/1+exp(-x))
w : weight in the neuron
b : bias in the neuron
y_pred : Input of the neuron

PROCESSES
forward : Feed an input(x) to the neuron, the neuron return the output(y_pred)
    0. x --> wx+b
    1. given wx+b = z, z-->sigma(z)
    2. sigma(z) --> y_pred

backpropagation : A process to adjust w and b in the neuron
    0. Generate training data of x[i] and y[i]
    1. Feed x[i] to the neuron, the neuron return the output(y_pred[i])
    2. Calculate 'square loss of y[i] and y_pred[i]'
       loss = (y[i]-y_pred[i])**2
    3. Calculate gradient of {loss} respects to {y_pred[i]}
       loss_grad[i] = 2 * (y[i]-y_pred[i]) * -1
    4. Calculate gradient of {y_pred[i]} respects to {z[i]}
       sigmoid_grad[i] = y_pred[i] * (1-y_pred[i])
    5. Calculate gradient of {z[i]} respects to {w}
       z_grad_w[i] = x[i]
    6. Calculate gradient of {z[i]} respects to {b}
       z_grad_b[i] = 1
    7. Calculate gradient of {loss} respects to {w}, using Chain rule
       w_grad[i] = loss_grad[i] * sigmoid_grad[i] * z_grad_w[i]
    8. Calculate gradient of {loss} respects to {b}, using Chain rule
       b_grad[i] = loss_grad[i] * sigmoid_grad[i] * 1
    9. Update w and b, using w_grad[i] and b_grad[i], given lr = learning rate
       w -= lr * w_grad[i]
       b -= lr * b_grad[i]
------------------------
日本語
------------------------
図
x -> [sigma(wx+b)] -> y_pred

記号
[] : ニューロン
x : ニューロンの入力
sigma : アクティベーション機能、シグモイド関数の使用 sigma(x) = (1/1+exp(-x))
w : ニューロンの重量
b : ニューロンのバイアス
y_pred : ニューロンの出力

プロセス
forward : 入力（x）をニューロンにフィードし、ニューロンは出力を返します（y_pred）
    0. x --> wx+b
    1. given wx+b = z, z-->sigma(z)
    2. sigma(z) --> y_pred
backpropagation : ニューロンの w と b を調整するプロセス
    0. x [i]とy [i]のトレーニングデータを生成する
    1. x [i]をニューロンにフィードすると、ニューロンは出力を返します（y_pred [i]）
    2. y [i]とy_pred [i]の「二乗損失」を計算する
       loss[i] = (y[i]-y_pred[i])**2
    3. {y_pred [i]}に対する{loss[i]}点の勾配を計算する
       loss_grad[i] = 2 * (y[i]-y_pred[i]) * -1
    4. {z [i]}に対する{y_pred [i]}の勾配の計算する
       sigmoid_grad[i] = y_pred[i] * (1-y_pred[i])
    5. {w}に対する{z [i]}の勾配の計算する
       z_grad_w[i] = x[i]
    6. {z [i]}の勾配を{b}に対して計算する
       z_grad_b[i] = 1
    7. 連鎖法則を使用して、{x}に対する{loss}の勾配を計算する
       w_grad[i] = loss_grad[i] * sigmoid_grad[i] * z_grad_w[i]
    8. 連鎖法則を使用して、{b}に対する{loss}の勾配を計算する
       b_grad[i] = loss_grad[i] * sigmoid_grad[i] * 1
    9. lr =学習率を指定して、w_grad [i]およびb_grad [i]を使用してwおよびbを更新する。
       w -= lr * w_grad[i]
       b -= lr * b_grad[i]
'''

import numpy as np

class Neuron:
    def __init__(self):
        np.random.seed(10)
        # weight in the neuron
        # ニューロンの重量
        self.w = np.random.randn()
        self.w_grad = None
        # bias in the neuron
        # ニューロンのバイアス
        self.b = np.random.randn()
        self.b_grad = None
        # learning rate
        # 学習率
        self.lr = 0.001
    def forward(self,x):
        return 1/(1+np.exp(-((self.w*x)+self.b)))
    def backpropagation(self,x,y,iteration):
        for i in range(iteration):
            y_pred = self.forward(x[i])
            loss = (y[i]-y_pred)**2
            # loss gradient respects to y_pred : -2*(y[i]-y_pred)
            # y_predに関する損失勾配：-2 *（y [i] -y_pred）
            loss_grad = -2*(y[i]-y_pred)
            # sigmoid(z)'s gradient = sigmoid(z)*(1-sigmoid(z))
            # sigmoid（z）の勾配= sigmoid（z）*（1-sigmoid（z））
            # y_pred(z) gradient respects to z : z = (wa+b)
            # y_pred（z）はzに対する勾配を考慮：z =（wa + b）
            sigmoid_grad = y_pred*(1-y_pred)
            # z gradient respects to w : a
            # wに対するz勾配の関係：a
            z_grad_w = x[i]
            # z gradient respects to b : wa
            # bに対するz勾配の関係：wa
            z_grad_b = 1
            # Chain rule: loss gradient respects to w = loss_grad*sigmoid_grad*z_grad
            # 連鎖規則：w = loss_grad * sigmoid_grad * z_gradに関する損失勾配
            self.w_grad = loss_grad*sigmoid_grad*z_grad_w
            # Chain rule: loss gradient respects to b = loss_grad*sigmoid_grad*1
            # 連鎖規則：b = loss_grad * sigmoid_grad * 1に関する損失勾配
            self.b_grad = loss_grad*sigmoid_grad*z_grad_b
            # Update w and b
            # wとbを更新
            self.w -= self.lr*self.w_grad
            self.b -= self.lr*self.b_grad

            # print results
            # 結果を印刷する
            if ((i+1)%10 == 0) or (i==iteration):
                print('Iteration {} loss {} w {} b {}'.format(i+1,loss,self.w,self.b))

# train on funtion y = 2x+3 : A Neuron can solve a linear function
# 関数 y = 2x + 3のトレーニング：ニューロンは線形関数を解くことができます
def y_func(x):
    return (2*x) + 3
x_train = []
y_train = []
# generate train data
# トレーニングデータを生成する
for i in range(10000):
    x_train.append(np.random.randn())
    y_train.append(y_func(x_train[-1]))

# Create Neuron
# ニューロンを作成する
A_Neuron = Neuron()
# Train Neuron
# ニューロンのトレーニング
A_Neuron.backpropagation(x_train,y_train,len(x_train))


