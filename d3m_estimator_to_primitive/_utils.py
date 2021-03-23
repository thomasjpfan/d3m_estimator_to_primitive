import numpy as np
from importlib import import_module
from inspect import signature
import inspect
from sklearn.base import is_regressor

from docstring_parser import parse
from sklearn.datasets import make_classification


def _get_class_split(class_name):
    class_name_split = class_name.split(".")

    module_str = ".".join(class_name_split[:-1])
    class_name_str = class_name_split[-1]
    return (module_str, class_name_str)


def _get_estimator_cls(class_name):
    class_name_split = class_name.split(".")

    module_str = ".".join(class_name_split[:-1])
    class_name_str = class_name_split[-1]
    module = import_module(module_str)

    Estimator = getattr(module, class_name_str)
    return Estimator


def _train_estimator(Estimator):
    X, y = make_classification(random_state=0)
    est = Estimator()
    est.fit(X, y)
    return est


def _convert_attribute_to_type(value):
    if isinstance(value, int):
        return "int"
    elif isinstance(value, float):
        return "float"
    elif isinstance(value, (tuple, str, list, dict)):
        return type(value).__qualname__
    elif isinstance(value, np.ndarray):
        return "ndarray"
    else:
        return "object"


def _convert_default_to_str(name, default):
    if isinstance(default, float):
        if np.isnan(default):
            return "np.nan"
        else:
            return f"{default}"
    elif isinstance(default, int):
        return f"{default}"
    elif isinstance(default, str):
        return f'"{default}"'
    elif default is None:
        return "None"
    else:
        raise ValueError(f"Unsupported default: {default} for {name}")


def _extract_metadata(estimator):
    docstring = parse(estimator.__doc__)
    init_params = signature(estimator.__class__).parameters

    param_meta = {}
    for name, param in init_params.items():
        if param.default == inspect.Parameter.empty:
            continue
        param_meta[name] = {"default": _convert_default_to_str(name, param.default)}

    for item in docstring.params:
        if item.arg_name not in param_meta:
            continue
        param_meta[item.arg_name]["description"] = item.description

    all_vars = vars(estimator)
    attributes = {
        key: _convert_attribute_to_type(value)
        for key, value in all_vars.items()
        if key not in param_meta
    }

    output = {"parameters": param_meta, "attributes": attributes}

    if is_regressor(estimator):
        output["primitive_family"] = "metadata_base.PrimitiveFamily.REGRESSION"
    else:
        output["primitive_family"] = "metadata_base.PrimitiveFamily.CLASSIFICATION"

    return output
