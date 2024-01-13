try:
    import cupy, cupyx
    USE_GPU = True
except Exception as e:
    if isinstance(e, ImportError):
        print("Cupy failed to import, using CPU (numpy & scipy)")
    else:
        print("Cupy caused error while importing: " + e)
    USE_GPU = False
