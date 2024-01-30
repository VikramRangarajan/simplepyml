r"""
SimplePyML

At this level, the choice of whether to use the GPU is made (`simplepyml/__init__`_).

.. _simplepyml/__init__: https://github.com/VikramRangarajan/simplepyml/blob/main/simplepyml/__init__.py

An attempt is made to import cupy and cupyx (for cupyx.scipy). 

If this attempt
fails, then numpy is used. Otherwise, cupy and cupyx.scipy and all other
GPU modules are used.

"""

try:
    import cupy, cupyx

    x = cupy.array([1, 2, 3])
    y = cupy.array([4, 5, 6])
    x + y
    USE_GPU = True
    print("Cupy successfully imported, using GPU")
except Exception as e:
    if isinstance(e, ImportError):
        print("Cupy failed to import, using CPU (numpy & scipy)")
    else:
        print("Cupy caused error while importing: " + e)
    USE_GPU = False
