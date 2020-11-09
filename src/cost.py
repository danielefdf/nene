# -*- coding: utf-8 -*-

from enum import Enum

import numpy as np


class Cost:  # aka loss function

    class Type(Enum):
        ABS_ERROR = 1
        QUADRATIC = 2

    @staticmethod
    def fn(t, a, y):

        if   t == Cost.Type.ABS_ERROR:
            return np.abs(a - y)

        elif t == Cost.Type.QUADRATIC:
            return np.linalg.norm(a - y)**2

    @staticmethod
    def dv(t, a, y):

        if   t == Cost.Type.ABS_ERROR:
            return 2 * np.heaviside((a - y), 1) - 1  # abs() derivative

        elif t == Cost.Type.QUADRATIC:
            return 2 * (a - y)






































































































