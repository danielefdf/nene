# nene
A neural network from scratch (only Numpy)

The elements of the project are as follows.

### elements

- activation.py: functions and derivatives for different types of activation function

    - STEP     # step function - perceptrons
    - LINEAR   # linear function
    - SIGMOID  # sigmoid
    - TANH     # tanh
    - RELU     # ReLU
    - L_RELU   # leaky ReLU

- cost.py: functions and derivatives for different types of activation function

    - ABS_ERROR  # absolute difference
    - QUADRATIC  # quadratic error

- datasetxy.py: functions for input selection

    - ALL     # all points of the plane
    - RANDOM  # random selected points from the plane

- initialization.py: functions for different types of initialization (for weights and biases)

    - ZERO      # initialization with zero
    - RANDOM    # random gaussian distribution
    - CONSTANT  # constant initialization

- regularization.py: functions for different types of regularization

    - NONE  # no regularization
    - L1    # L1 regularization
    - L2    # L2 regularization

- training.py: types of training

    - GRADIENT_D   # gradient descent
    - STOCHSTC_GD  # stochastic gradient descent

### monitor

- monitor.py: functions for displaying data from the neural network

### neural network

- neuralNetwork.py: the neural network, that uses all other elements

### test

- test.and.py: neural network representation of an AND logic gate
- test.classn.binflags.py: a classification for two values, with output as flags ([1,0] or [0,1])
- test.classn.binvalues.py: a classification for two values, with output as single values ([0] or [1])
- test.classn.example.py: a slightly more complex problem of classification
- test.clustg.autoencdr.py: an example of autoencoder neural network
- test.regrsn.py: a regression problem

