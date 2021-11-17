import warnings


def restrict_parallel(obj):
    warnings.warn(
        f"Parallel [{obj}] execution can lead to undefined behaviour", RuntimeWarning)
    return obj


def deprecate(obj):
    warnings.warn(f"{obj} is deprecated", DeprecationWarning)
    return obj


def warn_on_create(msg, warning_type=UserWarning):

    def cls_new_wrapper(cls):
        old_new_handler = cls.__init__

        def _init(obj, *args, **kwargs):
            warnings.warn(msg, warning_type)
            obj = old_new_handler(obj, *args, **kwargs)
            return obj

        setattr(cls, '__init__', _init)

        return cls

    return cls_new_wrapper
