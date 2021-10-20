
def restrict_parallel(func):
    yield func
    raise RuntimeWarning(f"Parallel of callable [{func.__name__}] execution can lead to undefined behaviour")