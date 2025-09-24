import numpy as np

# Simple linear regression example:
# We have input X and target y and want to learn parameters w (slope) and b (intercept)
# such that predictions ≈ y. We use mean squared error and gradient descent to update w and b.

X = np.array([1,2,3,4,5,6])  # input feature values (1D)
y = np.array([100,200,300,400,500,600])  # target values corresponding to each X

# initialize parameters (weights). Starting at zero is common for simple demos.
w = 0
b = 0

learning_rate = 0.01  # step size for gradient descent; too large can diverge, too small is slow
epochs = 3000  # number of passes over the data

for epoch in range(epochs):
    # predictions from the current model: y_hat = w * x + b
    predications = w * X + b

    # Compute gradients of the mean squared error loss with respect to parameters.
    # MSE loss L = (1/n) * sum((y_hat - y)^2)
    # dL/dw = (2/n) * sum((y_hat - y) * x)  -- factor 2 can be absorbed into learning rate
    # dL/db = (2/n) * sum(y_hat - y)
    # Here we omit the factor 2 (common simplification) and compute averaged gradients:
    dw = (1 / len(X)) * np.sum((predications - y) * X)  # gradient w.r.t. w (slope)
    db = (1 / len(X)) * np.sum((predications - y))      # gradient w.r.t. b (intercept)

    # Gradient descent parameter update: move parameters in the negative gradient direction
    # to reduce the loss.
    w -= learning_rate * dw
    b -= learning_rate * db

# After training, w and b should approximate the slope and intercept that fit the data.
print("Optimal parameters: \n w =", w, "\n b =", b)
