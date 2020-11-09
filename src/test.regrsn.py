# -*- coding: utf-8 -*-

import numpy  as np

#from datasetxy      import DatasetXY      as dd
from initialization import Initialization as ii
from activation     import Activation     as aa
from cost           import Cost           as cc
from regularization import Regularization as rr
from training       import Training       as tt
from neuralNetwork  import NeuralNetwork  as nn


if __name__ == '__main__':

    ''' dataset '''

    groups = (['sqrt'])

    def f(x):
        return np.power(x, 2).tolist()

    x_max = 11

    traing_x_list = []
    traing_y_list = []
    evaltn_x_list = []
    evaltn_y_list = []
    for x in range(0, x_max):
        traing_x_list.append([x])
        traing_y_list.append([f(x)])
        evaltn_x_list.append([x + 1/2])
        evaltn_y_list.append([f(x + 1/2)])

    ''' neural network '''

    layers = [['i'] * len(traing_x_list[0]),
              ['h'] * 3,
              ['h'] * 5,
              ['h'] * 3,
              ['o'] * len(traing_y_list[0])]

    actvtn_types     = [aa.Type.TANH for l in layers]
    actvtn_types[-1] = aa.Type.LINEAR  # ultimo nodo necessariamente lineare

    initn_weight_type = ii.WeightType.RANDOM
    initn_bias_type   = ii.BiasType.ZERO

    cost_type = cc.Type.ABS_ERROR

    reg_type   = rr.Type.L2
    reg_lambda = 0.001

    traing_type = tt.Type.GRADIENT_D
    epochs      = 50001   # di conseguenza, necessariamente alto
    batch_size  = 0       # len(traing_x)
    learng_eta  = 0.0001  # necessariamente basso

    input_log = False

    traing_qtts_log = True
    traing_acts_log = False
    traing_cost_log = False
    evaltn_cost_log = False

    n_net = nn(layers, actvtn_types,
            initn_weight_type, initn_bias_type,
            cost_type,
            reg_type, reg_lambda,
            traing_type, epochs, batch_size, learng_eta,
            groups,
            traing_x_list, traing_y_list,
            evaltn_x_list, evaltn_y_list,
            input_log,
            traing_qtts_log, traing_acts_log, traing_cost_log,
            evaltn_cost_log)

    n_net.training()
    n_net.predictions()




























































