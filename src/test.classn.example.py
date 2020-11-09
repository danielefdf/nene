# -*- coding: utf-8 -*-

#from datasetxy      import DatasetXY      as dd
from initialization import Initialization as ii
from activation     import Activation     as aa
from cost           import Cost           as cc
from regularization import Regularization as rr
from training       import Training       as tt
from neuralNetwork  import NeuralNetwork  as nn


if __name__ == '__main__':

    ''' dataset '''

    groups = ([0], [1])

    traing_x_list = [[0,0,0],[0,1,0],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
    traing_y_list = [[0],    [1],    [0],    [0],    [1],    [1]]

    evaltn_x_list = [[0,0,1],[0,1,1],[0,1,1]]
    evaltn_y_list = [[0],    [1],    [1]]

    # traing_data_ratio

    ''' neural network '''

    layers = [['i'] * len(traing_x_list[0]),
              ['h'] * 4,
              ['h'] * 4,
              ['o'] * len(traing_y_list[0])]

    # con sigmoid non va
    actvtn_types = [aa.Type.TANH for i in range(1,5)]

    initn_weight_type = ii.WeightType.RANDOM
    initn_bias_type   = ii.BiasType.ZERO
    cost_type         = cc.Type.QUADRATIC
    traing_type       = tt.Type.GRADIENT_D
    reg_type          = rr.Type.NONE
    reg_lambda        = 0.01
    learng_eta        = 0.03
    epochs            = 5001
    batch_size        = 0  # len(traing_x)

    input_log = True

    traing_qtts_log = False
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

















































