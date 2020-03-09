# Naive Bayes Algorithm

def normal_distribution(X):
    Mu = mean(X)
    Sigma = std(X)
    distribution = norm(Mu, Sigma)
    return distribution

X0 = X[y == 0]
X1 = X[y == 1]

prior0 = len(X0) / len(X)
prior1 = len(X1) / len(X)

X1_y0 = normal_distribution(X0[:, 0])
X2_y0 = normal_distribution(X0[:, 1])

X1_y1 = normal_distribution(X1[:, 0])
X2_y1 = normal_distribution(X1[:, 1])

def posteriorProb(X, prior, dist1, dist2):
    posterior = prior * dist1.pdf(X[0]) * dist2.pdf(X[1]) 
    return posterior
