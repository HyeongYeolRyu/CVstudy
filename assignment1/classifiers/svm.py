import numpy as np
from svm import *
from svmutil import *

class SVM:
    def __init__(self):
        pass
    
    def train(self, X_train, y_train, params=None):
        if params:
            svm_type, kernel_type, degree, C = params
        param = svm_parameter()
        param.svm_type = 0
        param.kernel_type = 1
        param.degree = 4
        param.C = 7
        prob = svm_problem(y_train, X_train)
        m = svm_train(prob, param)
        svm_save_model("trained_svm.xml", m)

    def predict(self, X, y):
        m = svm_load_model("trained_svm.xml")
        p_labs, p_acc, p_vals = svm_predict(y, X, m)
        print(p_labs)
        
        return p_labs