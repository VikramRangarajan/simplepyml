r"""
SimplePyML Activation Functions

These are simply functions that take in scalars or ndarrays
and return other scalars or ndarrays. They are also differentiable,
with the derivative being defined by the :py:func:`deriv` function

These objects are callable using :py:func:`__call__`, but are python
classes and objects. You must declare an Activation object and call
the object as such::

    x = np.array([-1, 0, 1])
    linear_obj = Relu()
    y = linear_obj(x) # Gives you np.array([0, 0, 1])

"""
