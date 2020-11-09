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

- initialization.py: functions for different types of initialization (weights and biases)

    weights
        - ZERO    # initialization with zero
        - RANDOM  # random gaussian distribution
    biases
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

- test.and.py: ???
- test.classn.binflags.py: ???
- test.classn.binvalues.py: ???
- test.classn.example.py: ???
- test.clustg.autoencdr.py: ???
- test.regrsn.py: ???

