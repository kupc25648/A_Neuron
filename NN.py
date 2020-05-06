'''
========================
A BASIC NEURON
========================
'''
'''
x -> [sigma(wx+b)] -> y_pred
SYMBOLS
[] : Neuron
x : Input of the neuron
sigma : activation funtion, using sigmoid function sigma(x) = (1/1+exp(-x))
w : weight in the neuron
b : bias in the neuron
y_pred : Output of the neuron

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
'''
import numpy as np

class Neuron:
    def __init__(self):
        np.random.seed(10)
        self.w = np.random.randn()
        self.w_grad = None
        self.b = np.random.randn()
        self.b_grad = None
        self.lr = 0.001
    def forward(self,x):
        return 1/(1+np.exp(-((self.w*x)+self.b)))
    def backpropagation(self,x,y,iteration):
        for i in range(iteration):
            y_pred = self.forward(x[i])
            loss = (y[i]-y_pred)**2
            #loss gradient respects to y_pred : -2*(y[i]-y_pred)
            loss_grad = -2*(y[i]-y_pred)
            # sigmoid(z)'s gradient = sigmoid(z)*(1-sigmoid(z))
            #y_pred(z) gradient respects to z : z = (wa+b)
            sigmoid_grad = y_pred*(1-y_pred)
            # z gradient respects to w : a
            z_grad_w = x[i]
            # z gradient respects to b : wa
            z_grad_b = 1
            # Chain rule: loss gradient respects to w = loss_grad*sigmoid_grad*z_grad
            self.w_grad = loss_grad*sigmoid_grad*z_grad_w
            # Chain rule: loss gradient respects to b = loss_grad*sigmoid_grad*1
            self.b_grad = loss_grad*sigmoid_grad*z_grad_b
            # Update
            self.w -= self.lr*self.w_grad
            self.b -= self.lr*self.b_grad

            #show result
            if i%10 == 0:
                print('Iteration {} loss {} w {} b {}'.format(i,loss,self.w,self.b))

# test on funtion y = 2x+3 : A Neuron can solve a linear function
def y_func(x):
    return (2*x) + 3
x_train = []
y_train = []
# generate test data
for i in range(10000):
    x_train.append(np.random.randn())
    y_train.append(y_func(x_train[-1]))

# Create Neuron
A_Neuron = Neuron()
# Train Neuron
A_Neuron.backpropagation(x_train,y_train,len(x_train))












