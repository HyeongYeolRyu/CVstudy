import numpy as np
from svm import *
from svmutil import *

class SVM:
    def __init__(self):
        pass
    
    def train(self, X_train, y_train):
        #X_train = X_train.tolist()
        param = svm_parameter()
        prob = svm_problem(y_train, X_train)
        m = svm_train(prob, param)
        svm_save_model("trained_svm.xml", m)

    def predict(self, X, y):
        #X = X.tolist()
        m = svm_load_model("trained_svm.xml")
        p_labs, p_acc, p_vals = svm_predict(y, X, m)
        print(p_labs)
        
        return p_labs