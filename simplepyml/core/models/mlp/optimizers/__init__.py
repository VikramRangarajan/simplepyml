r"""
SimplePyML MLP Optimizers Module

This module contains the available optimizers, which
include but are not limited to:

* Stoichastic Gradient Descent (SGD)
* Adam

More will hopefully be added (ex: Root Mean Square Error).

These optimizers implement the :py:class:`.Optimizer`
superclass

The organization of these optimizers needs change, but right now,
the all implement the :py:func:`__call__` function with training
data. The :py:func:`run_epoch` function is used to run an epoch,
and this is called in a loop in the :py:func:`__call__`.

Batching has not yet been implemented, and is one of my goals.
This will be a large feature as it requires a restructure of
every single layer, as they currently only support 1 input at
a time.

"""
