import numpy as np
import matplotlib.pyplot as plt

class DecisionBoundary:
    def __init__(self, data, scale):
        self.data = data
        self.scale = scale

    def generate_grid(self):
        xmax = max(self.data[:,0])+self.scale
        xmin = min(self.data[:,0])-self.scale
        ymax = max(self.data[:,1])+self.scale
        ymin = min(self.data[:,1])-self.scale


        x = np.arange(xmin, xmax, .03)
        y = np.arange(ymin, ymax, .03)

        X_grid, Y_grid = np.meshgrid(x,y)
        return [X_grid, Y_grid]

    def flatten(self, grided_xg, grided_yg):
        X_ = np.vstack((grided_xg.flatten(), grided_yg.flatten())).T
        return X_
    
    def classs_labels(self, predicted_targets, label):
        res_g = []
        for val in predicted_targets:
            if val >= 0.5:
                val = 1
                res_g.append(val)
            else:
                val = -1
                res_g.append(val)
        res_g = np.array(res_g)
        cl1 = np.where(res_g == 1)
        cl2 = np.where(res_g == -1)
        return cl1, cl2

    def ploting(self, data_labels, c1, c2, X_):
        plt.plot(X_[c1][:, 0], X_[c1][:, 1], c='cyan', marker=(8, 2, 0), linewidth=0.5)
        plt.plot(X_[c2][:, 0], X_[c2][:, 1], c='pink', marker=(8, 2, 0), linewidth=0.5)
        plt.plot(self.data[:, 0:1][data_labels == 1], self.data[:, 1:2][data_labels == 1], 
        c='b', marker='x', linestyle='none', markersize=5)
        plt.plot(self.data[:, 0:1][data_labels == -1], self.data[:, 1:2][data_labels == -1], 
        c='r', marker='o', linestyle='none', markersize=5)
        plt.show()