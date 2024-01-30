r"""
MLP (Multi-Layer Perceptron) Network

The `MLP`_ consists of layers such that the output of layer n is
the input of layer n+1. An input to the network is sent through
all layers and the output of the last layer is the output of the network
for that input.

.. _MLP: https://en.wikipedia.org/wiki/Multilayer_perceptron

To train this network, backpropagation is used and several 
:py:mod:`.optimizers` are available

There are also different :py:mod:`.layers` layers to choose
from. More are to be added.

"""
