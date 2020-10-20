import numpy as np
from numpy.linalg import eig

# number of variables
n=3
# number of observations
p=10

np.random.seed(1)

X = np.random.normal(5.0, 1.0, size=(p, n))

# mean center the observation
bar_X = X.mean(axis=0)
Y = X-bar_X

# numpy covariance takes the unusual format which expected n x p for n variables p observations
# actually taking covariance on the original X also fine. 
# But using Y is easier to understand geomatrically.
C = np.cov(Y.transpose())

# eigendecomosition
eig_vals, U = eig(C)
Lambda = np.diag(eig_vals)

# check eigendecomposition correctness
# NOTE: when we are running huge amount of real life data points, this can really be unmatched!!!
print('Covariance can be reconstructed:', np.all(np.isclose(C-U.dot(Lambda).dot(U.transpose()), 0)))

# the explained variance of each principal components
print('Explained Variance:', eig_vals/np.sum(eig_vals))

# Projection of Y data points into Principal Components (i.e. scores)
W = Y.dot(U)

# Reconstruction of the original data
X_new = W.dot(U.transpose()) + bar_X

# Check data reconstruction result
print('Reconstructed data matched:', np.all(np.isclose(X - X_new,0)))