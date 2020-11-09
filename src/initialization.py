# -*- coding: utf-8 -*-

from enum import Enum

import numpy as np


class Initialization:

    class WeightType(Enum):
        ZERO   = 1  # zero initialization
        RANDOM = 2  # random gaussian distribution

    class BiasType(Enum):
        ZERO     = 1  # zero initialization
        RANDOM   = 2  # random gaussian distribution
        CONSTANT = 3  # constant initialization

    @staticmethod
    def set_weights(t, xSize):

        if   t == Initialization.WeightType.ZERO:
            return 0

        elif t == Initialization.WeightType.RANDOM:
            return np.random.uniform(size=xSize)

    @staticmethod
    def set_biases(t, xSize):

        if   t == Initialization.BiasType.ZERO:
            return 0

        elif t == Initialization.BiasType.RANDOM:
            return np.random.uniform(size=xSize)





























































































