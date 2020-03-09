# Logistic Regression

theta = np.zeros((X.shape[1], 1))

def sigmoid_function(X):
    sigmoid = 1 / (1 + np.exp(-X))
    return sigmoid

def net_input(theta, X):
    return np.dot(X, theta)

def probability(theta, x):
    # Returns the probability after passing through sigmoid
    return sigmoid(net_input(theta, x))

def loss_function(theta, X, y):
    # Computes the cost function for all the training samples
    m = X.shape[0]
    total_cost = -(1 / m) * np.sum(
        y * np.log(probability(theta, X)) + (1 - y) * np.log(
            1 - probability(theta, X)))
    return total_cost

def gradient(theta, X, y):
    # Computes the gradient of the cost function at the point theta
    m = X.shape[0]
    return (1 / m) * np.dot(X.T, sigmoid(net_input(theta,   X)) - y)


def find_weights( X, y, theta, lr, tol):
    
    while True:  
        
        if(gradient(theta, X, y) < tol):
            break

        dtheta = gradient(theta, X, y)
        theta = theta - lr*dtheta
    
    return theta

