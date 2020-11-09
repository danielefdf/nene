# -*- coding: utf-8 -*-

import random as rm
import numpy  as np

from initialization import Initialization as ii
from activation     import Activation     as aa
from cost           import Cost           as cc
from regularization import Regularization as rr
from training       import Training       as tt
from monitor        import Monitor        as mm


class NeuralNetwork(object):

    def __init__(self, layers, actvtn_types,
            initn_weight_type, initn_bias_type,
            cost_type,
            reg_type, reg_lambda,
            traing_type, epochs, batch_size, learng_eta,
            groups,
            traing_x_list, traing_y_list, evaltn_x_list, evaltn_y_list,
            input_log,
            traing_qtts_log,
            traing_acts_log, traing_cost_log, evaltn_cost_log):

        ''' NN features '''

        self.layers            = layers
        self.actvtn_types      = actvtn_types
        self.initn_weight_type = initn_weight_type
        self.initn_bias_type   = initn_bias_type
        self.cost_type         = cost_type

        self.reg_type   = reg_type
        self.reg_lambda = reg_lambda

        ''' groups '''

        self.groups = groups

        ''' training features '''

        self.traing_type = traing_type
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.learng_eta  = learng_eta

        ''' training data '''

        self.traing_x = np.array(traing_x_list)
        self.traing_y = np.array(traing_y_list)

        ''' evaluation data '''

        self.evaltn_x = np.array(evaltn_x_list)
        self.evaltn_y = np.array(evaltn_y_list)

        ''' model : list of arrays '''

        self.l_number = len(self.layers)

        self.l_sizes   = []
        self.l_weights = []
        self.l_biases  = []
        self.l_actvtns = []
        for l_ix in range(0, self.l_number):
            self.l_sizes.append(len(self.layers[l_ix]))
            self.l_weights.append(np.array([]))
            self.l_biases.append(np.array([]))
            self.l_actvtns.append(np.array([]))

        # weights and biases initialization
        for l_ix in range(1, self.l_number):
            w_size = (self.l_sizes[l_ix-1], self.l_sizes[l_ix])
            b_size = (self.l_sizes[l_ix])
            self.l_weights[l_ix] = ii.set_weights(self.initn_weight_type,
                    w_size) / len(self.traing_x)  # scaling
            self.l_biases[l_ix]  = ii.set_biases(self.initn_bias_type,
                    b_size)

        ''' monitor features '''

        self.input_log       = input_log
        self.traing_qtts_log = traing_qtts_log
        self.traing_acts_log = traing_acts_log
        self.traing_cost_log = traing_cost_log
        self.evaltn_cost_log = evaltn_cost_log

        self.monitor = mm(self)

        ''' show input '''

        if self.input_log:
            if self.traing_x[0].shape == (2,):
                self.monitor.show_2d_input_map()

    def feed_forward(self, x):  # aka forward propagation

        l_actvtns = x
        for l_ix in range(1, self.l_number):
            l_values = np.dot(l_actvtns, self.l_weights[l_ix]) \
                    + self.l_biases[l_ix]
            l_actvtns = aa.fn(self.actvtn_types[l_ix], l_values)
            self.l_actvtns[l_ix] = l_actvtns

    def back_propagtn(self, x, y):

        learng_coeff = self.learng_eta / len(x)  # scaling

        l_losses = cc.dv(self.cost_type, y, self.l_actvtns[self.l_number-1]) \
                + rr.dv(self.reg_type, y, self.l_actvtns[self.l_number-1])

        for l_ix in range(self.l_number - 1, 0, -1):
            l_slopes = aa.dv(self.actvtn_types[l_ix], self.l_actvtns[l_ix])
            l_deltas = l_losses * l_slopes
            weights_nabla = self.l_actvtns[l_ix-1].T.dot(l_deltas)
            biases_nabla  = np.sum(l_deltas, axis=0, keepdims=True)[0]
            self.l_weights[l_ix] = self.l_weights[l_ix] \
                    + weights_nabla * learng_coeff
            self.l_biases[l_ix]  = self.l_biases[l_ix] \
                    + biases_nabla * learng_coeff
            l_losses = l_deltas.dot(self.l_weights[l_ix].T)

    def gradient_descent(self):

        self.l_actvtns[0] = self.traing_x

        for self.epoch in range(self.epochs):
            self.feed_forward(self.traing_x)
            self.back_propagtn(self.traing_x, self.traing_y)
            self.monitor.check_append()

    def stochastic_gradient_descent(self):

        data_indexes = []
        for d_ix in range(len(self.traing_x)):
            data_indexes.append(d_ix)
        rm.shuffle(data_indexes)

        for self.epoch in range(self.epochs):
            for k in range(0, len(self.traing_x), self.batch_size):
                batchX = self.traing_x[data_indexes[k:k+self.batch_size]]
                batchY = self.traing_y[data_indexes[k:k+self.batch_size]]
                self.l_actvtns[0] = batchX
                self.feed_forward(batchX)
                self.back_propagtn(batchX, batchY)
            self.monitor.check_append()

    def training(self):

        if   self.traing_type == tt.Type.GRADIENT_D:
            self.gradient_descent()

        elif self.traing_type == tt.Type.STOCHSTC_GD:
            self.stochastic_gradient_descent()

        self.monitor.show_output()

    def prediction(self, x):

        l_actvtns = x
        for l_ix in range(1, self.l_number):
            l_values = np.dot(l_actvtns, self.l_weights[l_ix]) \
                    + self.l_biases[l_ix]
            l_actvtns = aa.fn(self.actvtn_types[l_ix], l_values)

        return l_actvtns

    def predictions(self):

        self.traing_predictions = self.prediction(self.traing_x)
        
        if len(self.evaltn_x) > 0:
            self.evaltn_predictions = self.prediction(self.evaltn_x)

        self.monitor.show_predictions()















































