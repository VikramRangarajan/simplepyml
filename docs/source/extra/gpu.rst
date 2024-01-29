Using the GPU
=============

SimplePyML Uses `cupy`_ in order to access GPU accelerated machine learning.

.. _cupy: https://cupy.dev/

This was done for simplicity reasons, and also the point of this project was to avoid
using any pre-existing ML libraries (`TensorFlow`_, `PyTorch`_), so I would not be able to use
tensors from those libraries (`torch.Tensor`_, `tf.Tensor`_).

.. _TensorFlow: https://tensorflow.org/
.. _PyTorch: https://pytorch.org/
.. _torch.Tensor: https://pytorch.org/docs/stable/tensors.html
.. _tf.Tensor: https://www.tensorflow.org/api_docs/python/tf/Tensor

In addition, these libraries have support for `automatic differentiation`_. My goal is to
calculate these backpropagation formulas by hand, or even eventually implement my own
autograd library (unlikely, since the formulas for things like convolutions and
operations over certain axes sounds incredibly difficult).

.. _automatic differentiation: https://en.wikipedia.org/wiki/Automatic_differentiation