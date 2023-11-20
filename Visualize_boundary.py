import numpy as np
import matplotlib.pyplot as plt
from Classify_vis import DecisionBoundary
from MLP_numpy import MLP_NeuralNetwork
from scipy import io

syn = io.loadmat('synthetic')
X = syn['X']
Y = syn['Y']
train = np.array(X)
labels = np.array(Y)

nn = MLP_NeuralNetwork(input=2, hidden=4, output=1)
training = nn.train(train, labels, iterations=30, N=0.1)

decision = DecisionBoundary(X, 0.5)
grided_xg, grided_yg = decision.generate_grid()
X_ = decision.flatten(grided_xg, grided_yg)
yp_g = np.array(nn.predict(X_))
cl1, cl2 = decision.classs_labels(yp_g, Y)
decision.ploting(Y, cl1, cl2, X_)
