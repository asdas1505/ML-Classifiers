def estimate_W_linear_regression_pinv(X, y):
    
    pinv = np.linalg.pinv(np.dot(X.T,X))
    W = np.dot(pinv, np.transpose(X))
    W = np.dot(W, y)
    
    return W
