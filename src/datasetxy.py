# -*- coding: utf-8 -*-

from enum import Enum

import numpy as np


class DatasetXY:

    class XType(Enum):
        ALL    = 1  # all points of the plane
        RANDOM = 2  # random (gaussian distribution) selected points

    def __init__(self, t, x_lims, groups, traing_data_ratio, f):

        self.x_lims = x_lims
        self.groups = groups

        self.traing_data_ratio = traing_data_ratio

        ''' setting x '''

        self.x_range = range(self.x_lims[0], self.x_lims[1])
        self.x_range_len = self.x_lims[1] - self.x_lims[0]
        self.x_range_area = self.x_range_len ** 2

        if   t == DatasetXY.XType.ALL:
            x = np.array([[[r, c] for c in self.x_range] \
                    for r in self.x_range])
            x = x.reshape((self.x_range_area, 2))

        elif t == DatasetXY.XType.RANDOM:
            x = np.random.randint(self.x_lims[0], self.x_lims[1], \
                    (self.x_range_area, 2))

        self.x_list = x.tolist()

        ''' setting y '''

        self.y_list = [f(e) for e in self.x_list]

        ''' splitting training and evaluation data '''

        self.traing_x_list = []
        self.traing_y_list = []
        self.evaltn_x_list = []
        self.evaltn_y_list = []
        for e in zip(self.x_list, self.y_list):
            e_x = e[0]
            e_y = e[1]
            if np.random.randint(1, self.traing_data_ratio + 1) == 1:
                self.evaltn_x_list.append(e_x)
                self.evaltn_y_list.append(e_y)
            else:
                self.traing_x_list.append(e_x)
                self.traing_y_list.append(e_y)




















































































