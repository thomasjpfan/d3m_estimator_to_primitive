from contextlib import suppress
from ._utils import _get_class_split

TUNING_PARAM_URL = "https://metadata.datadrivendiscovery.org/types/TuningParameter"
CONTROL_PARAM_URL = "https://metadata.datadrivendiscovery.org/types/ControlParameter"
RESOURCE_PARAM_URL = (
    "https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter"
)
semantic_type_to_url = {
    "tuning": TUNING_PARAM_URL,
    "control": CONTROL_PARAM_URL,
    "resource": RESOURCE_PARAM_URL,
}


def _process_description(description):
    desp_list = description.split("\n")
    desp_list = [f'"{item}"' for item in desp_list if item]
    desp_joined = "\n".join(desp_list)
    return f"({desp_joined})"


def _get_semantic_types(user_hp_info):
    semantic_urls = []
    semantic_types = user_hp_info.get("semantic_types", ["tuning"])
    for semantic_type in semantic_types:
        with suppress(KeyError):
            url = semantic_type_to_url[semantic_type]
            semantic_urls.append(f"'{url}'")

    joined_urls = ",".join(semantic_urls)
    return f"[{joined_urls}]"


def _create_d3m_hyperparam(d3m_dtype, d3m_kwargs):
    output_str = f"{d3m_dtype}("
    for key, value in d3m_kwargs.items():
        output_str += f"{key}={value}, "

    output_str += ")"
    return output_str


def _process_union_child(child_output):
    output_str = "OrderedDict({"
    for _, (new_name, child_d3m) in child_output.items():
        output_str += f'"{new_name}": {child_d3m}, '

    output_str += "})"
    return output_str


def _process_union_type(name, user_hp_info, trained_hp_info):
    d3m_dtype = "hyperparams.Union"

    child_output = {}
    for choice in user_hp_info["choices"]:
        choice_type = choice["type"]
        new_name = f"{name}_{choice_type}"
        output = _process_hp_metedata(new_name, choice, {})
        child_output[choice_type] = (new_name, output)

    default = child_output[user_hp_info["default_type"]][0]
    hyper_kwargs = {"default": f'"{default}"'}

    hyper_kwargs["configuration"] = _process_union_child(child_output)

    with suppress(KeyError):
        hyper_kwargs["description"] = _process_description(
            trained_hp_info["description"]
        )

    hyper_kwargs["semantic_types"] = _get_semantic_types(user_hp_info)
    return _create_d3m_hyperparam(d3m_dtype, hyper_kwargs)


def _process_hp_metedata(name, user_hp_info, trained_hp_info):

    if user_hp_info["type"] == "float":
        d3m_dtype = "hyperparams.Bounded[float]"
    elif user_hp_info["type"] == "int":
        d3m_dtype = "hyperparams.Bounded[int]"
    elif user_hp_info["type"] == "bool":
        d3m_dtype = "hyperparams.UniformBool"
    elif user_hp_info["type"] == "union":
        return _process_union_type(name, user_hp_info, trained_hp_info)
    elif user_hp_info["type"] == "enum":
        d3m_dtype = "hyperparams.Enumeration[str]"
    elif user_hp_info["type"] == "None":
        d3m_dtype = "hyperparams.Constant"
    else:
        raise ValueError(f"Unknown metadata for {name}")

    hyper_kwargs = {}
    with suppress(KeyError):
        hyper_kwargs["description"] = _process_description(
            trained_hp_info["description"]
        )

    if d3m_dtype.startswith("hyperparams.Bounded"):
        hyper_kwargs["lower"] = user_hp_info.get("lower", None)
        hyper_kwargs["upper"] = user_hp_info.get("upper", None)

        other_list = ["lower_inclusive", "upper_inclusive"]
        for name in other_list:
            with suppress(KeyError):
                hyper_kwargs[name] = user_hp_info[name]
    elif user_hp_info["type"] == "None":
        hyper_kwargs["default"] = "None"

    # special handling for enumeration
    if d3m_dtype == "hyperparams.Enumeration[str]":
        values = [f'"{item}"' for item in user_hp_info["choices"]]
        value_joined = ",".join(values)
        hyper_kwargs["values"] = f"[{value_joined}]"

    # defaults
    if "default" in user_hp_info:
        hyper_kwargs["default"] = user_hp_info["default"]
    elif "default" in trained_hp_info:
        hyper_kwargs["default"] = trained_hp_info["default"]

    hyper_kwargs["semantic_types"] = _get_semantic_types(user_hp_info)

    return _create_d3m_hyperparam(d3m_dtype, hyper_kwargs)


def _prepare_for_templating(estimator_name, user_metadata, trained_metadata):
    output = {"estimator_name": estimator_name}

    output_parmas = {}

    # custom import
    module_str, class_name_str = _get_class_split(user_metadata["class_name"])
    output["custom_imports"] = [f"from {module_str} import {class_name_str}"]
    output["class_name_single"] = class_name_str
    output["class_name_complete"] = user_metadata["class_name"]

    # Params
    learned_attributes = trained_metadata.get("attributes", {})
    for name, attr_type in learned_attributes.items():
        output_parmas[name] = attr_type

    output["params"] = output_parmas

    # Hyperparameters
    trained_parameters = trained_metadata["parameters"]
    hyperparameters = user_metadata["parameters"]

    output_hyperparameters = {}
    for name, user_hp_info in hyperparameters.items():
        trained_hp_info = trained_parameters.get(name, {})
        output_hyperparameters[name] = _process_hp_metedata(
            name, user_hp_info, trained_hp_info
        )
    output["hyperparams"] = output_hyperparameters

    # hyperparams_to_tune
    with suppress(KeyError):
        hyperparams_to_tune = user_metadata["hyperparams_to_tune"]
        hyperparams_to_tune = [f'"{tune}"' for tune in hyperparams_to_tune]
        hyper_join = ",".join(hyperparams_to_tune)
        output["hyperparams_to_tune"] = f"[{hyper_join}]"

    output["has_random_state"] = "random_state" in trained_metadata["parameters"]
    output["d3m_python_path"] = user_metadata["d3m_python_path"]

    algorithm_types = [
        f"metadata_base.PrimitiveAlgorithmType.{algo_type}"
        for algo_type in user_metadata["algorithm_types"]
    ]
    algorithm_types = ",".join(algorithm_types)
    output["algorithm_types"] = f"[{algorithm_types}]"

    return output
