# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pp
import matplotlib.cm as cm

from cost import Cost as cc


class Monitor(object):

    FIG_SIZE = (6, 6)
    AX_RECT = [0.1, 0.1, 0.8, 0.8]
    DECIMALS_ANNOT = 4
    SHOW_ANNOTATIONS = False
    PROGRESSIVE_SELF_PLOT = True
    PRED_SIGNIF_FIG = 3

    def __init__(self, n_net):

        self.n_net = n_net

        self.traing_acts = []

        self.traing_costs  = []
        self.traing_accies = []

        self.evaltn_costs  = []
        self.evaltn_accies = []

        if self.n_net.traing_qtts_log:
            self.self_figure = pp.figure(figsize=Monitor.FIG_SIZE)
            self.self_figure.canvas.mpl_connect(
                    'key_press_event', self.on_key_press)
            self.self_axes_log = []
            self.self_axes_ix = 0

        if self.n_net.traing_acts_log:
            acts_figure = pp.figure(figsize=Monitor.FIG_SIZE)
            self.acts_ax = acts_figure.add_axes(Monitor.AX_RECT,
                    title='trend', label='trend',
                    xticks=[], xticklabels=[])

        if (self.n_net.traing_cost_log
                or self.n_net.evaltn_cost_log):
            cost_figure = pp.figure(figsize=Monitor.FIG_SIZE)
            self.cost_ax = cost_figure.add_axes(Monitor.AX_RECT,
                    title='cost', label='cost',
                    xticks=[], xticklabels=[])

    def show_2d_input_map(self):

        if self.n_net.input_log:
            self.inpt_figure = pp.figure(figsize=Monitor.FIG_SIZE)
            self.inpt_ax = self.inpt_figure.add_axes(Monitor.AX_RECT,
                    title='input')

        def lighter(color):
            return color/2 + 0.3

        colors = cm.rainbow(np.linspace(0, 1, len(self.n_net.groups)))
        colors = [[lighter(c)] for c in colors]

        # training data

        data = []
        for g in self.n_net.groups:
            x_sel = np.array([e[0] for e in \
                    zip(self.n_net.traing_x, self.n_net.traing_y) \
                    if all(e[1] == g)])
            data.append(x_sel)

        edgec = 'white'

        for dat1, color, group in zip(data, colors, self.n_net.groups):
            for e in dat1:
                x, y = e
                self.inpt_ax.scatter(x, y, alpha=0.8, s=30, c=color,
                        edgecolors=edgec, label=group)

        # evaluation data

        data = []
        for g in self.n_net.groups:
            x_sel = np.array([e[0] for e in \
                    zip(self.n_net.evaltn_x, self.n_net.evaltn_y) \
                    if all(e[1] == g)])
            data.append(x_sel)

        edgec = 'black'

        for dat1, color, group in zip(data, colors, self.n_net.groups):
            for e in dat1:
                x, y = e
                self.inpt_ax.scatter(x, y, alpha=0.8, s=30, c=color,
                        edgecolors=edgec, label=group)

    def total_cost(self, inputs, outputs):

        cost = 0.
        for (x, y) in zip(inputs, outputs):
            predctn = self.n_net.prediction(x)
            cost += cc.fn(self.n_net.cost_type, predctn, y) \
                    / len(x)  # scaling

        return cost

    def edge_width(self, weight):

        ''' calcolo max(weights) '''

        # List(Array) to Array(Array) con np.concatenate()
        flat_w = np.concatenate(self.n_net.l_weights, axis=None).flatten()
        w_max = np.max(flat_w)

        ''' calcolo dimensione '''

        # parte assoluta (con tanh per limitare dimensione a 1)
        w_abs = np.tanh(np.abs(weight))

        # parte relativa (sempre compresa tra 0 e 1)
        w_rel = w_abs / w_max

        # mix: riflette la dimensione assoluta e relativa
        w_mixed = (1/2) * (w_abs + w_rel)

        # applico un coefficiente arbitrario per la massima ampiezza
        coeff = 20

        return coeff * w_mixed

    def append_self_ax(self):

        ax_rect = Monitor.AX_RECT
        ax_title = 'epoch: ' + str(self.n_net.epoch)
        ax_label = ax_title  # to avoid deprecation warning
        self.ax = self.self_figure.add_axes(ax_rect, visible=False,
                title=ax_title, label=ax_label)
        self.ax.axis('off')

        left = .1; right = .9; bottom = .1; top = 1.1
        v_spacing = (top - bottom) / float(max(self.n_net.l_sizes))
        h_spacing = (right - left) / float(self.n_net.l_number - 1)

        # neurons
        for l_ix, l_length in enumerate(self.n_net.l_sizes):
            lTop = v_spacing * (l_length - 1.) / 2. + (top - bottom) / 2.
            for n_ix in range(l_length):
                n_type = self.n_net.layers[l_ix][n_ix]
                c_x = l_ix * h_spacing + left
                c_y = lTop - n_ix * v_spacing
                c_radius = v_spacing / 6.
                if   n_type == 'i': c_color = 'white';       c_ec = 'blue'
                elif n_type == 'h': c_color = 'aquamarine';  c_ec = 'green'
                elif n_type == 'o': c_color = 'greenyellow'; c_ec = 'black'
                circle = pp.Circle(xy=(c_x, c_y), radius=c_radius,
                                   color=c_color, ec=c_ec, zorder=4)
                self.ax.add_artist(circle)
                if self.SHOW_ANNOTATIONS:
                    # values annotations
                    a_x = c_x - c_radius
                    a_y = c_y - c_radius * 1.6
                    a_v = np.round(a=self.n_net.l_actvtns[l_ix].T[n_ix],
                            decimals=Monitor.DECIMALS_ANNOT)
                    self.ax.annotate(a_v, xy=(a_x, a_y))
                    # biases annotations
                    a_x = c_x - c_radius
                    a_y = c_y + c_radius * 1.5
                    if len(self.n_net.l_biases[l_ix]) == 0:
                        a_v = '[]'
                    else:
                        a_v = np.round(a=self.n_net.l_biases[l_ix][n_ix],
                                decimals=Monitor.DECIMALS_ANNOT)
                    self.ax.annotate(a_v, xy=(a_x, a_y))

        # edges
        lengths_enum = enumerate(
                zip(self.n_net.l_sizes[:-1], self.n_net.l_sizes[1:]))
        for l_ix, (l_length_from, l_length_to) in lengths_enum:
            top_from = v_spacing * (l_length_from - 1.) / 2. \
                    + (top - bottom) / 2.
            top_to   = v_spacing * (l_length_to - 1.) / 2. \
                    + (top - bottom) / 2.
            for l_from_ix in range(l_length_from):
                for l_to_ix in range(l_length_to):
                    # weight data
                    weight = self.n_net.l_weights[l_ix+1][l_from_ix][l_to_ix]
                    # weight line
                    from_x = l_ix * h_spacing + left
                    to_x   = (l_ix + 1) * h_spacing + left
                    from_y = top_from - l_from_ix * v_spacing
                    to_y   = top_to - l_to_ix * v_spacing
                    l_color = 'palegreen'
                    l_width = self.edge_width(weight)
                    line = pp.Line2D(xdata=[from_x,to_x], ydata=[from_y,to_y],
                            color=l_color, linewidth=l_width)
                    self.ax.add_artist(line)
                    if self.SHOW_ANNOTATIONS:
                        # weight annotations
                        a_x = from_x + c_radius * 1.5
                        a_y = from_y - v_spacing / 5. * l_to_ix
                        a_v = np.round(a=weight,
                                decimals=Monitor.DECIMALS_ANNOT)
                        self.ax.annotate(a_v, xy=(a_x, a_y))

        self.self_axes_log.append(self.ax)

    def append(self):

        if self.n_net.traing_qtts_log:
            if not self.PROGRESSIVE_SELF_PLOT:
                self.append_self_ax()

        if self.n_net.traing_acts_log:
            acts = self.n_net.l_actvtns[self.n_net.l_number-1].tolist()
            self.traing_acts.append(acts)

        if self.n_net.traing_cost_log:
            cost = self.total_cost(self.n_net.traing_x, self.n_net.traing_y)
            self.traing_costs.append(cost)

        if self.n_net.evaltn_cost_log:
            cost = self.total_cost(self.n_net.evaltn_x, self.n_net.evaltn_y)
            self.evaltn_costs.append(cost)

    def check_append(self):

        epoch = self.n_net.epoch
        epochs = self.n_net.epochs

        if self.n_net.traing_qtts_log:
            if self.PROGRESSIVE_SELF_PLOT:
                b = 4
                for exp in range(2, 100):
                    '''
                    per b = 10:
                    if (   (self.epoch <   100 and self.epoch %   0 == 0)
                        or (self.epoch <  1000 and self.epoch %  10 == 0)
                        or (self.epoch < 10000 and self.epoch % 100 == 0)
                        or (...)
                        or (self.epoch == self.epochs)):
                    '''
                    if ((epoch < b ** exp and epoch % (b ** (exp - 2)) == 0)
                            or epoch == epochs):
                        self.append_self_ax()
                        break

        b = 400
        if (epoch % b == 0
                or epoch == epochs):
            self.append()

    def next_self_plot(self):

        if self.self_axes_ix < len(self.self_axes_log) - 1:
            self.self_axes_log[self.self_axes_ix].set_visible(False)
            self.self_axes_log[self.self_axes_ix+1].set_visible(True)
            self.self_axes_ix += 1
            self.self_figure.canvas.draw()

    def prev_self_plot(self):

        if self.self_axes_ix > 0:
            self.self_axes_log[self.self_axes_ix].set_visible(False)
            self.self_axes_log[self.self_axes_ix-1].set_visible(True)
            self.self_axes_ix -= 1
            self.self_figure.canvas.draw()

    def on_key_press(self, event):

        if   event.key == 'right': self.next_self_plot()
        elif event.key == 'left':  self.prev_self_plot()
        else:
            return

    def show_net_quantities_plot(self):
        self.self_axes_log[0].set_visible(True)

    def show_traing_acts_plot(self):
        for vIx in range(0, len(self.n_net.traing_x)):
            a_list = [ta[vIx] for ta in self.traing_acts]
            self.acts_ax.plot(np.array(a_list))

    def show_traing_cost_plot(self):
        self.cost_ax.plot(np.array(self.traing_costs))

    def show_evaltn_cost_plot(self):
        self.cost_ax.plot(np.array(self.evaltn_costs))

    def show_output(self):

        if self.n_net.traing_qtts_log:
            self.show_net_quantities_plot()

        if self.n_net.traing_acts_log:
            self.show_traing_acts_plot()

        if self.n_net.traing_cost_log:
            self.show_traing_cost_plot()

        if self.n_net.evaltn_cost_log:
            self.show_evaltn_cost_plot()

    def show_predictions(self):

        print('')
        print('predictions on training data')

        for (inpt, outt, pred) in zip(self.n_net.traing_x,
                self.n_net.traing_y, self.n_net.traing_predictions):
            perf = abs(outt - pred)
            print(self.prediction_string(inpt, outt, pred, perf))

        if (self.n_net.evaltn_x.size == 0):
            return

        print('')
        print('predictions on evaluation data')

        for (inpt, outt, pred) in zip(self.n_net.evaltn_x,
                self.n_net.evaltn_y, self.n_net.evaltn_predictions):
            perf = abs(outt - pred)
            print(self.prediction_string(inpt, outt, pred, perf))

    def prediction_string(self, inpt, outt, pred, perf):

        float_lim_str = "{:+" + str(self.PRED_SIGNIF_FIG + 1 + 3) + ".2f}"
        int_lim_str   = "{:+" + str(self.PRED_SIGNIF_FIG + 1)     +    "}"

        np.set_printoptions(formatter={
                'float_kind': float_lim_str.format,
                  'int_kind': int_lim_str.format})

        inpt_str = str(np.round(inpt, 2))
        outt_str = str(np.round(outt, 2))
        pred_str = str(np.round(pred, 2))

        inpt_str = inpt_str.replace('+', ' ')
        outt_str = outt_str.replace('+', ' ')
        pred_str = pred_str.replace('+', ' ')

        sum_perf = int(np.sum(np.abs(perf)))

        if (sum_perf * 10 <= 5):
            perf_str = '*' * (sum_perf * 10)
        else:
            perf_str = '* (' + str(sum_perf) + ')'

        return inpt_str + "    " + outt_str + "    " + pred_str + "    " \
                + perf_str
































































