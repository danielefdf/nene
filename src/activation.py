# -*- coding: utf-8 -*-

from enum import Enum

import numpy as np


class Activation:

    class Type(Enum):
        STEP    = 1  # step function - perceptrons
        LINEAR  = 2
        SIGMOID = 3  # sigmoid(x)    := 1 / (1 + exp(-x)) -- aka logistic
        TANH    = 4  # tanh(x)       := (1 - exp(-2 * x)) / (1 + exp(-2 * x))
        RELU    = 5  # ReLU(x)       := (x > 0 ? x : 0)
        L_RELU  = 6  # leaky ReLU(x) := (x > 0 ? x : 0.01 * x)

    @staticmethod
    def fn(activation_type, x):

        if   activation_type == Activation.Type.STEP:
            return np.heaviside(x, 0)

        elif activation_type == Activation.Type.LINEAR:
            return x

        elif activation_type == Activation.Type.SIGMOID:
            return 1 / (1 + np.exp(-x))

        elif activation_type == Activation.Type.TANH:
            return np.tanh(x)

        elif activation_type == Activation.Type.RELU:
            return np.maximum(x, 0)

        elif activation_type == Activation.Type.L_RELU:
            return np.maximum(0.01 * x, x)

    @staticmethod
    def dv(activation_type, act_x):

        if   activation_type == Activation.Type.STEP:
            return 0

        elif activation_type == Activation.Type.LINEAR:
            return 1

        elif activation_type == Activation.Type.SIGMOID:
            return act_x * (1 - act_x)

        elif activation_type == Activation.Type.TANH:
            return 1 - act_x ** 2

        elif activation_type == Activation.Type.RELU:
            return np.maximum(0, np.heaviside(act_x, 0))

        elif activation_type == Activation.Type.L_RELU:
            return np.maximum(act_x, np.heaviside(act_x, 0))



























































