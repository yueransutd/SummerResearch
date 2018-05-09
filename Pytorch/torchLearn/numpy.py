import numpy as np

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension
N, D_in, H, D_out = 64,1000, 100,10

# Create random input and output data
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# Randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

lr = 1e-6
for t in range(500):
    # forward pass
    h = x.dot(w1)
    h_relu = np.maximun(h, 0)
    y_pred = h_relu.dot(w2)

    loss = np.square(y_pred - y).sum()
    print(t, loss)

    #backprop
    grad_y_pred = 2* (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h<0] = 0
    grad_w1 = x.T.dot(grad_h)

    #update weights
    w1 -= lr* grad_w1
    w2 -= lr* grad_w2

