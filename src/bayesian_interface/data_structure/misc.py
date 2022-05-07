from copy import deepcopy


def dynamically_load_class(module_name: str, class_name: str) -> object.__class__:
    """dynamically load class from specified module
    :param module_name: module name
    :param class_name: class name
    :return: loaded class object
    """
    import importlib

    foo = importlib.import_module(module_name)
    return getattr(foo, class_name)
