from typing import Any, Dict, List, Optional, Sequence
from numpy import ndarray
from collections import OrderedDict
from scipy import sparse
import sklearn
import numpy
import pandas

from xgboost import XGBClassifier

from d3m.container import DataFrame as d3m_dataframe
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m.base import utils as base_utils
from d3m.exceptions import PrimitiveNotFittedError
from d3m.primitive_interfaces.base import CallResult, DockerContainer

from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase


Inputs = d3m_dataframe
Outputs = d3m_dataframe


class Params(params.Params):
    n_estimators: Optional[int]
    max_depth: Optional[object]
    learning_rate: Optional[object]
    verbosity: Optional[object]
    booster: Optional[object]
    tree_method: Optional[object]
    gamma: Optional[object]
    min_child_weight: Optional[object]
    max_delta_step: Optional[object]
    subsample: Optional[object]
    colsample_bytree: Optional[object]
    colsample_bylevel: Optional[object]
    colsample_bynode: Optional[object]
    reg_alpha: Optional[object]
    reg_lambda: Optional[object]
    scale_pos_weight: Optional[object]
    base_score: Optional[object]
    missing: Optional[float]
    num_parallel_tree: Optional[object]
    kwargs: Optional[dict]
    random_state: Optional[object]
    n_jobs: Optional[object]
    monotone_constraints: Optional[object]
    interaction_constraints: Optional[object]
    importance_type: Optional[str]
    gpu_id: Optional[object]
    validate_parameters: Optional[object]
    classes_: Optional[ndarray]
    n_classes_: Optional[int]
    _le: Optional[object]
    _features_count: Optional[int]
    n_features_in_: Optional[int]
    _Booster: Optional[object]
    input_column_names: Optional[pandas.core.indexes.base.Index]
    target_names_: Optional[Sequence[Any]]
    training_indices_: Optional[Sequence[int]]
    target_column_indices_: Optional[Sequence[int]]
    target_columns_metadata_: Optional[List[OrderedDict]]


class Hyperparams(hyperparams.Hyperparams):
    n_estimators = hyperparams.Bounded[int](
        lower=1,
        upper=1000,
        default=20,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
    )
    use_inputs_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description=(
            "A set of column indices to force primitive to use as training input."
            " If any specified column cannot be parsed, it is skipped."
        ),
    )
    use_outputs_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description=(
            "A set of column indices to force primitive to use as training target."
            " If any specified column cannot be parsed, it is skipped."
        ),
    )
    exclude_inputs_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description=(
            "A set of column indices to not use as training inputs. "
            'Applicable only if "use_columns" is not provided.'
        ),
    )
    exclude_outputs_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description=(
            "A set of column indices to not use as training target. "
            'Applicable only if "use_columns" is not provided.'
        ),
    )
    return_result = hyperparams.Enumeration(
        values=["append", "replace", "new"],
        default="new",
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description=(
            "Should parsed columns be appended, should they replace original columns, "
            "or should only parsed columns be returned? This hyperparam is ignored if "
            "use_semantic_types is set to false."
        ),
    )
    use_semantic_types = hyperparams.UniformBool(
        default=False,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description=(
            "Controls whether semantic_types metadata will be used for filtering "
            "columns in input dataframe. Setting this to false makes the code ignore "
            "return_result and will produce only the output dataframe"
        ),
    )
    add_index_columns = hyperparams.UniformBool(
        default=False,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description=(
            "Also include primary index columns if input data has them. "
            'Applicable only if "return_result" is set to "new".'
        ),
    )
    error_on_no_input = hyperparams.UniformBool(
        default=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description=(
            "Throw an exception if no input column is selected/provided. "
            "Defaults to true to behave like sklearn. To prevent pipelines from "
            "breaking set this to False."
        ),
    )

    return_semantic_type = hyperparams.Enumeration[str](
        values=[
            "https://metadata.datadrivendiscovery.org/types/Attribute",
            "https://metadata.datadrivendiscovery.org/types/ConstructedAttribute",
            "https://metadata.datadrivendiscovery.org/types/PredictedTarget",
        ],
        default="https://metadata.datadrivendiscovery.org/types/PredictedTarget",
        description="Decides what semantic type to attach to generated output",
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
    )


class D3M_XGBClassifier(
    SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]
):
    """
    Primitive wrapping for xgboost.XGBClassifier
    """

    __author__ = "Thomas J. Fan"
    metadata = metadata_base.PrimitiveMetadata(
        {
            "name": "xgboost.XGBClassifier",
            "algorithm_types": [metadata_base.PrimitiveAlgorithmType.GRADIENT_BOOSTING],
            "python_path": "d3m.primitives.classification.xgboost.XGBClassifier",
            "version": "2021.3.0",
            "id": "f18653f7-7546-3630-9af3-55d72ea9000c",
            "primitive_family": metadata_base.PrimitiveFamily.CLASSIFICATION,
        }
    )

    def __init__(
        self,
        *,
        hyperparams: Hyperparams,
        random_seed: int = 0,
        docker_containers: Dict[str, DockerContainer] = None
    ) -> None:

        super().__init__(
            hyperparams=hyperparams,
            random_seed=random_seed,
            docker_containers=docker_containers,
        )

        self._clf = XGBClassifier(
            n_estimators=self.hyperparams["n_estimators"],
        )
        self._inputs = None
        self._outputs = None
        self._training_inputs = None
        self._training_outputs = None
        self._target_names = None
        self._training_indices = None
        self._target_column_indices = None
        self._target_columns_metadata: List[OrderedDict] = None
        self._input_column_names = None
        self._fitted = False
        self._new_training_data = False

    def get_params(self) -> Params:
        if not self._fitted:
            return Params(
                n_estimators=None,
                max_depth=None,
                learning_rate=None,
                verbosity=None,
                booster=None,
                tree_method=None,
                gamma=None,
                min_child_weight=None,
                max_delta_step=None,
                subsample=None,
                colsample_bytree=None,
                colsample_bylevel=None,
                colsample_bynode=None,
                reg_alpha=None,
                reg_lambda=None,
                scale_pos_weight=None,
                base_score=None,
                missing=None,
                num_parallel_tree=None,
                kwargs=None,
                random_state=None,
                n_jobs=None,
                monotone_constraints=None,
                interaction_constraints=None,
                importance_type=None,
                gpu_id=None,
                validate_parameters=None,
                classes_=None,
                n_classes_=None,
                _le=None,
                _features_count=None,
                n_features_in_=None,
                _Booster=None,
                input_column_names=self._input_column_names,
                training_indices_=self._training_indices,
                target_names_=self._target_names,
                target_column_indices_=self._target_column_indices,
                target_columns_metadata_=self._target_columns_metadata,
            )

        return Params(
            n_estimators=getattr(self._clf, "n_estimators", None),
            max_depth=getattr(self._clf, "max_depth", None),
            learning_rate=getattr(self._clf, "learning_rate", None),
            verbosity=getattr(self._clf, "verbosity", None),
            booster=getattr(self._clf, "booster", None),
            tree_method=getattr(self._clf, "tree_method", None),
            gamma=getattr(self._clf, "gamma", None),
            min_child_weight=getattr(self._clf, "min_child_weight", None),
            max_delta_step=getattr(self._clf, "max_delta_step", None),
            subsample=getattr(self._clf, "subsample", None),
            colsample_bytree=getattr(self._clf, "colsample_bytree", None),
            colsample_bylevel=getattr(self._clf, "colsample_bylevel", None),
            colsample_bynode=getattr(self._clf, "colsample_bynode", None),
            reg_alpha=getattr(self._clf, "reg_alpha", None),
            reg_lambda=getattr(self._clf, "reg_lambda", None),
            scale_pos_weight=getattr(self._clf, "scale_pos_weight", None),
            base_score=getattr(self._clf, "base_score", None),
            missing=getattr(self._clf, "missing", None),
            num_parallel_tree=getattr(self._clf, "num_parallel_tree", None),
            kwargs=getattr(self._clf, "kwargs", None),
            random_state=getattr(self._clf, "random_state", None),
            n_jobs=getattr(self._clf, "n_jobs", None),
            monotone_constraints=getattr(self._clf, "monotone_constraints", None),
            interaction_constraints=getattr(self._clf, "interaction_constraints", None),
            importance_type=getattr(self._clf, "importance_type", None),
            gpu_id=getattr(self._clf, "gpu_id", None),
            validate_parameters=getattr(self._clf, "validate_parameters", None),
            classes_=getattr(self._clf, "classes_", None),
            n_classes_=getattr(self._clf, "n_classes_", None),
            _le=getattr(self._clf, "_le", None),
            _features_count=getattr(self._clf, "_features_count", None),
            n_features_in_=getattr(self._clf, "n_features_in_", None),
            _Booster=getattr(self._clf, "_Booster", None),
            input_column_names=self._input_column_names,
            training_indices_=self._training_indices,
            target_names_=self._target_names,
            target_column_indices_=self._target_column_indices,
            target_columns_metadata_=self._target_columns_metadata,
        )

    def set_params(self, *, params: Params) -> None:
        self._clf.n_estimators = params["n_estimators"]
        self._clf.max_depth = params["max_depth"]
        self._clf.learning_rate = params["learning_rate"]
        self._clf.verbosity = params["verbosity"]
        self._clf.booster = params["booster"]
        self._clf.tree_method = params["tree_method"]
        self._clf.gamma = params["gamma"]
        self._clf.min_child_weight = params["min_child_weight"]
        self._clf.max_delta_step = params["max_delta_step"]
        self._clf.subsample = params["subsample"]
        self._clf.colsample_bytree = params["colsample_bytree"]
        self._clf.colsample_bylevel = params["colsample_bylevel"]
        self._clf.colsample_bynode = params["colsample_bynode"]
        self._clf.reg_alpha = params["reg_alpha"]
        self._clf.reg_lambda = params["reg_lambda"]
        self._clf.scale_pos_weight = params["scale_pos_weight"]
        self._clf.base_score = params["base_score"]
        self._clf.missing = params["missing"]
        self._clf.num_parallel_tree = params["num_parallel_tree"]
        self._clf.kwargs = params["kwargs"]
        self._clf.random_state = params["random_state"]
        self._clf.n_jobs = params["n_jobs"]
        self._clf.monotone_constraints = params["monotone_constraints"]
        self._clf.interaction_constraints = params["interaction_constraints"]
        self._clf.importance_type = params["importance_type"]
        self._clf.gpu_id = params["gpu_id"]
        self._clf.validate_parameters = params["validate_parameters"]
        self._clf.classes_ = params["classes_"]
        self._clf.n_classes_ = params["n_classes_"]
        self._clf._le = params["_le"]
        self._clf._features_count = params["_features_count"]
        self._clf.n_features_in_ = params["n_features_in_"]
        self._clf._Booster = params["_Booster"]
        self._input_column_names = params["input_column_names"]
        self._training_indices = params["training_indices_"]
        self._target_names = params["target_names_"]
        self._target_column_indices = params["target_column_indices_"]
        self._target_columns_metadata = params["target_columns_metadata_"]

        if params["n_estimators"] is not None:
            self._fitted = True
        if params["max_depth"] is not None:
            self._fitted = True
        if params["learning_rate"] is not None:
            self._fitted = True
        if params["verbosity"] is not None:
            self._fitted = True
        if params["booster"] is not None:
            self._fitted = True
        if params["tree_method"] is not None:
            self._fitted = True
        if params["gamma"] is not None:
            self._fitted = True
        if params["min_child_weight"] is not None:
            self._fitted = True
        if params["max_delta_step"] is not None:
            self._fitted = True
        if params["subsample"] is not None:
            self._fitted = True
        if params["colsample_bytree"] is not None:
            self._fitted = True
        if params["colsample_bylevel"] is not None:
            self._fitted = True
        if params["colsample_bynode"] is not None:
            self._fitted = True
        if params["reg_alpha"] is not None:
            self._fitted = True
        if params["reg_lambda"] is not None:
            self._fitted = True
        if params["scale_pos_weight"] is not None:
            self._fitted = True
        if params["base_score"] is not None:
            self._fitted = True
        if params["missing"] is not None:
            self._fitted = True
        if params["num_parallel_tree"] is not None:
            self._fitted = True
        if params["kwargs"] is not None:
            self._fitted = True
        if params["random_state"] is not None:
            self._fitted = True
        if params["n_jobs"] is not None:
            self._fitted = True
        if params["monotone_constraints"] is not None:
            self._fitted = True
        if params["interaction_constraints"] is not None:
            self._fitted = True
        if params["importance_type"] is not None:
            self._fitted = True
        if params["gpu_id"] is not None:
            self._fitted = True
        if params["validate_parameters"] is not None:
            self._fitted = True
        if params["classes_"] is not None:
            self._fitted = True
        if params["n_classes_"] is not None:
            self._fitted = True
        if params["_le"] is not None:
            self._fitted = True
        if params["_features_count"] is not None:
            self._fitted = True
        if params["n_features_in_"] is not None:
            self._fitted = True
        if params["_Booster"] is not None:
            self._fitted = True

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        self._inputs = inputs
        self._outputs = outputs
        self._fitted = False
        self._new_training_data = True

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        if self._inputs is None or self._outputs is None:
            raise ValueError("Missing training data.")

        if not self._new_training_data:
            return CallResult(None)
        self._new_training_data = False

        self._training_inputs, self._training_indices = self._get_columns_to_fit(
            self._inputs, self.hyperparams
        )
        (
            self._training_outputs,
            self._target_names,
            self._target_column_indices,
        ) = self._get_targets(self._outputs, self.hyperparams)
        self._input_column_names = self._training_inputs.columns.astype(str)

        if len(self._training_indices) > 0 and len(self._target_column_indices) > 0:
            self._target_columns_metadata = self._get_target_columns_metadata(
                self._training_outputs.metadata, self.hyperparams
            )
            sk_training_output = self._training_outputs.values

            shape = sk_training_output.shape
            if len(shape) == 2 and shape[1] == 1:
                sk_training_output = numpy.ravel(sk_training_output)

            self._clf.fit(self._training_inputs, sk_training_output)
            self._fitted = True
        else:
            if self.hyperparams["error_on_no_input"]:
                raise RuntimeError("No input columns were selected")
            self.logger.warn("No input columns were selected")

        return CallResult(None)

    def produce(
        self, *, inputs: Inputs, timeout: float = None, iterations: int = None
    ) -> CallResult[Outputs]:
        sk_inputs, columns_to_use = self._get_columns_to_fit(inputs, self.hyperparams)
        output = []
        if len(sk_inputs.columns):
            try:
                sk_output = self._clf.predict(sk_inputs)
            except sklearn.exceptions.NotFittedError as error:
                raise PrimitiveNotFittedError("Primitive not fitted.") from error
            # For primitives that allow predicting without fitting like GaussianProcessRegressor
            if not self._fitted:
                raise PrimitiveNotFittedError("Primitive not fitted.")
            if sparse.issparse(sk_output):
                sk_output = pandas.DataFrame.sparse.from_spmatrix(sk_output)
            output = self._wrap_predictions(inputs, sk_output)
            output.columns = self._target_names
            output = [output]
        else:
            if self.hyperparams["error_on_no_input"]:
                raise RuntimeError("No input columns were selected")
            self.logger.warn("No input columns were selected")
        outputs = base_utils.combine_columns(
            return_result=self.hyperparams["return_result"],
            add_index_columns=self.hyperparams["add_index_columns"],
            inputs=inputs,
            column_indices=self._target_column_indices,
            columns_list=output,
        )
        return CallResult(outputs)

    @classmethod
    def _get_columns_to_fit(cls, inputs: Inputs, hyperparams: Hyperparams):
        if not hyperparams["use_semantic_types"]:
            return inputs, list(range(len(inputs.columns)))

        inputs_metadata = inputs.metadata

        def can_produce_column(column_index: int) -> bool:
            return cls._can_produce_column(inputs_metadata, column_index, hyperparams)

        columns_to_produce, columns_not_to_produce = base_utils.get_columns_to_use(
            inputs_metadata,
            use_columns=hyperparams["use_inputs_columns"],
            exclude_columns=hyperparams["exclude_inputs_columns"],
            can_use_column=can_produce_column,
        )
        return inputs.iloc[:, columns_to_produce], columns_to_produce
        # return columns_to_produce

    @classmethod
    def _can_produce_column(
        cls,
        inputs_metadata: metadata_base.DataMetadata,
        column_index: int,
        hyperparams: Hyperparams,
    ) -> bool:
        column_metadata = inputs_metadata.query(
            (metadata_base.ALL_ELEMENTS, column_index)
        )

        accepted_structural_types = (int, float, numpy.integer, numpy.float64)
        accepted_semantic_types = set()
        accepted_semantic_types.add(
            "https://metadata.datadrivendiscovery.org/types/Attribute"
        )
        if not issubclass(
            column_metadata["structural_type"], accepted_structural_types
        ):
            return False

        semantic_types = set(column_metadata.get("semantic_types", []))

        if len(semantic_types) == 0:
            cls.logger.warning("No semantic types found in column metadata")
            return False
        # Making sure all accepted_semantic_types are available in semantic_types
        if len(accepted_semantic_types - semantic_types) == 0:
            return True

        return False

    @classmethod
    def _get_targets(cls, data: d3m_dataframe, hyperparams: Hyperparams):
        if not hyperparams["use_semantic_types"]:
            return data, list(data.columns), list(range(len(data.columns)))

        metadata = data.metadata

        def can_produce_column(column_index: int) -> bool:
            accepted_semantic_types = set()
            accepted_semantic_types.add(
                "https://metadata.datadrivendiscovery.org/types/TrueTarget"
            )
            column_metadata = metadata.query((metadata_base.ALL_ELEMENTS, column_index))
            semantic_types = set(column_metadata.get("semantic_types", []))
            if len(semantic_types) == 0:
                cls.logger.warning("No semantic types found in column metadata")
                return False
            # Making sure all accepted_semantic_types are available in semantic_types
            if len(accepted_semantic_types - semantic_types) == 0:
                return True
            return False

        (
            target_column_indices,
            target_columns_not_to_produce,
        ) = base_utils.get_columns_to_use(
            metadata,
            use_columns=hyperparams["use_outputs_columns"],
            exclude_columns=hyperparams["exclude_outputs_columns"],
            can_use_column=can_produce_column,
        )
        targets = []
        if target_column_indices:
            targets = data.select_columns(target_column_indices)
        target_column_names = []
        for idx in target_column_indices:
            target_column_names.append(data.columns[idx])
        return targets, target_column_names, target_column_indices

    @classmethod
    def _get_target_columns_metadata(
        cls, outputs_metadata: metadata_base.DataMetadata, hyperparams
    ) -> List[OrderedDict]:
        outputs_length = outputs_metadata.query((metadata_base.ALL_ELEMENTS,))[
            "dimension"
        ]["length"]

        target_columns_metadata: List[OrderedDict] = []
        for column_index in range(outputs_length):
            column_metadata = OrderedDict(outputs_metadata.query_column(column_index))

            # Update semantic types and prepare it for predicted targets.
            semantic_types = set(column_metadata.get("semantic_types", []))
            semantic_types_to_remove = set(
                [
                    "https://metadata.datadrivendiscovery.org/types/TrueTarget",
                    "https://metadata.datadrivendiscovery.org/types/SuggestedTarget",
                ]
            )
            add_semantic_types = set(
                [
                    "https://metadata.datadrivendiscovery.org/types/PredictedTarget",
                ]
            )
            add_semantic_types.add(hyperparams["return_semantic_type"])
            semantic_types = semantic_types - semantic_types_to_remove
            semantic_types = semantic_types.union(add_semantic_types)
            column_metadata["semantic_types"] = list(semantic_types)

            target_columns_metadata.append(column_metadata)

        return target_columns_metadata

    @classmethod
    def _update_predictions_metadata(
        cls,
        inputs_metadata: metadata_base.DataMetadata,
        outputs: Optional[Outputs],
        target_columns_metadata: List[OrderedDict],
    ) -> metadata_base.DataMetadata:
        outputs_metadata = metadata_base.DataMetadata().generate(value=outputs)

        for column_index, column_metadata in enumerate(target_columns_metadata):
            column_metadata.pop("structural_type", None)
            outputs_metadata = outputs_metadata.update_column(
                column_index, column_metadata
            )

        return outputs_metadata

    def _wrap_predictions(self, inputs: Inputs, predictions: ndarray) -> Outputs:
        outputs = d3m_dataframe(predictions, generate_metadata=False)
        outputs.metadata = self._update_predictions_metadata(
            inputs.metadata, outputs, self._target_columns_metadata
        )
        return outputs

    @classmethod
    def _add_target_columns_metadata(cls, outputs_metadata: metadata_base.DataMetadata):
        outputs_length = outputs_metadata.query((metadata_base.ALL_ELEMENTS,))[
            "dimension"
        ]["length"]

        target_columns_metadata: List[OrderedDict] = []
        for column_index in range(outputs_length):
            column_metadata = OrderedDict()
            semantic_types = []
            semantic_types.append(
                "https://metadata.datadrivendiscovery.org/types/PredictedTarget"
            )
            column_name = outputs_metadata.query(
                (metadata_base.ALL_ELEMENTS, column_index)
            ).get("name")
            if column_name is None:
                column_name = "output_{}".format(column_index)
            column_metadata["semantic_types"] = semantic_types
            column_metadata["name"] = str(column_name)
            target_columns_metadata.append(column_metadata)

        return target_columns_metadata


D3M_XGBClassifier.__doc__ = XGBClassifier
