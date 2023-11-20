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
res = nn.predict(X)
res = np.array(res)[:, -1]
ress = []
for val in res:
    if val >= 0.5:
        val = 1
        ress.append(val)
    else:
        val = -1
        ress.append(val)
ress = np.array(ress)
a = np.where(ress == Y[:, -1])
a = np.array(a)
print("accuracy is :", a.shape[1]/len(Y))
c1 = np.where(np.array(ress) == 1)
c2 = np.where(np.array(ress) == -1)
class1 = X[:, 0:1][c1]
class2 = X[:, 0:1][c2]

plt.figure()


plt.subplot(1, 1, 1)
plt.scatter(X[:, 0:1][Y==1], X[:, 1:2][Y==1], c='r')
plt.scatter(X[:, 0:1][Y==-1], X[:, 1:2][Y==-1], c='b')
plt.title('Ground Truth')
plt.legend()
plt.show()

# plt.subplot(1, 2, 2)
# plt.scatter(class1, X[:, 1:2][c1], c='r')
# plt.scatter(class2, X[:, 1:2][c2], c='b')
# plt.title('After Training')
# plt.legend()
# plt.show()