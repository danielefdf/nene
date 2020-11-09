# -*- coding: utf-8 -*-

from enum import Enum

import numpy as np


'''
lasso regression -- Least Absolute Shrinkage and Selection Operator
ridge Regression -- ridge = cresta
'''

class Regularization:

    class Type(Enum):
        NONE = 0
        L1   = 1  # aka lasso regression
        L2   = 2  # aka ridge regression

    '''
    @staticmethod
    def fn(t, a, y):

        if   t == Regularization.Type.NONE:
            return 0

        elif t == Regularization.Type.L1:
            return np.abs(a - y)

        elif t == Regularization.Type.L2:
            return (a - y) ** 2
    '''

    @staticmethod
    def dv(t, a, y):

        if   t == Regularization.Type.NONE:
            return 0

        elif t == Regularization.Type.L1:
            return 2 * np.heaviside((a - y), 1) - 1  # abs() derivative

        elif t == Regularization.Type.L2:
            return 2 * (a - y)





































































































