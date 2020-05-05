'''
========================
A BASIC NEURON
========================
'''
'''
x -> sigma(wx+b) -> y
    forward
    backpropagation
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
            #loss gradient respects to y_pred : 2*(y[i]-y_pred)
            loss_grad = 2*(y[i]-y_pred)
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
            self.w += self.lr*self.w_grad
            self.b += self.lr*self.b_grad

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












