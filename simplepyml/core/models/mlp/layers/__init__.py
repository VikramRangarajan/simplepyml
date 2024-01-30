r"""
SimplePyML MLP Layers Module

This module contains all the available layers for the MLP
network

These layers are inspired by the `keras layers api`_.

They all use lazy initialization (the first time the layer
is called using an array, the input shape and other
parameters are set from that array)

.. _keras layers api: https://keras.io/api/layers/

All of these layers implement the python :py:func:`__call__` function
so allow you to call a layer object like a function.

All layers also implement the :py:class:`.Layer` class,
which implement a :py:func:`backgrad` function.

 
Given :math:`\frac{\partial L}{\partial Y}`, :py:func:`backgrad`
will calculate the loss gradient w.r.t. each layer parameter (e.g.
weights, kernels, biases, etc.) and also the gradient w.r.t. the
input.

This is how I have implemented backpropagation.

"""
