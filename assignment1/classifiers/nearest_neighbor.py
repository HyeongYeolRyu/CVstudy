import numpy as np

class NearestNeighbor:
    def __init__(self, X, y):
        self.Xtr = X
        self.ytr = y
    
    def compute_distance(self, X_train, X_test):
        return np.sum(np.square(X_train - X_test), axis = 1)
        
    def predict(self, X_test):
        num = X_test.shape[0]
        y_pred = []
        for i in range(num):
            dist = self.compute_distance(self.Xtr, X_test[i])
            y_pred.append(self.ytr[np.argmin(dist)])
        y_pred = np.array(y_pred)
        
        return y_pred