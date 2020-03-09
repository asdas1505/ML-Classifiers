def estimate_W_linear_regression_GD(X, y, tol, learning_rate):
    
    while True:
        
        if(np.linalg.norm(loss) < tol):
            break
            
        W =  np.random.rand(y.shape(0), X.shape(1))         # Initializing W randomly 
        ypred = np.dot(W,X)
        loss = np.dot((y -ypred).T, y-ypred)
        dW = -2*np.dot(X.T, y-ypred)
        W = W - learning_rate*dW
        if(np.linalg.norm(dW) < tol):
            break
    return W            
    
