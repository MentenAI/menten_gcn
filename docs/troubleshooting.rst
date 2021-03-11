===============
Troubleshooting
===============

***********
Sparse Mode
***********

1. Cannot convert a symbolic Tensor to a numpy array

.. code-block::

    NotImplementedError: Cannot convert a symbolic Tensor (gradient_tape/model_1/crystal_conv/sub:0) to a numpy array. This error may indicate that you're trying to pass a Tensor to a NumPy call, which is not supported
    
This is a tough one to debug because it is often thrown from deep inside keras.
It appears to be solved by updating versions of python, numpy, and/or tensorflow.
For example, we might see this pop up in python 3.7 but not 3.6 or 3.8.
`This stack overflow question <https://stackoverflow.com/questions/58479556/notimplementederror-cannot-convert-a-symbolic-tensor-2nd-target0-to-a-numpy>`_
also suggests that the numpy/tensorflow relationship could be fixed by downgrading numpy below 1.2.
