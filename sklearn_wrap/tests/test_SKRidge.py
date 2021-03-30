import unittest
import pickle

from sklearn_wrap import SKRidge
from pathlib import Path
from d3m.metadata import base as metadata_base
from d3m import container
from d3m.primitive_interfaces.base import PrimitiveBase
from d3m.exceptions import PrimitiveNotFittedError
from pandas.testing import assert_frame_equal

from sklearn_wrap.vendor.common_primitives import dataset_to_dataframe, column_parser

dataset_doc_path = (
    Path(__file__).parent.parent
    / "tests-data"
    / "datasets"
    / "iris_dataset_1"
    / "datasetDoc.json"
)
dataset_doc_path = dataset_doc_path.resolve()

dataset = container.Dataset.load(dataset_uri="file://{}".format(dataset_doc_path))
hyperparams_class = dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.query()[
    "primitive_code"
]["class_type_arguments"]["Hyperparams"]
primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(
    hyperparams=hyperparams_class.defaults()
)
call_metadata = primitive.produce(inputs=dataset)

dataframe = call_metadata.value
column_parser_htperparams = column_parser.Hyperparams.defaults()
column_parser_primitive = column_parser.ColumnParserPrimitive(
    hyperparams=column_parser_htperparams
)
parsed_dataframe = column_parser_primitive.produce(inputs=dataframe).value
parsed_dataframe.metadata = parsed_dataframe.metadata.add_semantic_type(
    (metadata_base.ALL_ELEMENTS, 5),
    "https://metadata.datadrivendiscovery.org/types/Target",
)
parsed_dataframe.metadata = parsed_dataframe.metadata.add_semantic_type(
    (metadata_base.ALL_ELEMENTS, 5),
    "https://metadata.datadrivendiscovery.org/types/TrueTarget",
)
parsed_dataframe.metadata = parsed_dataframe.metadata.remove_semantic_type(
    (metadata_base.ALL_ELEMENTS, 5),
    "https://metadata.datadrivendiscovery.org/types/Attribute",
)
parsed_dataframe.metadata = parsed_dataframe.metadata.add_semantic_type(
    (metadata_base.ALL_ELEMENTS, 5),
    "https://metadata.datadrivendiscovery.org/types/CategoricalData",
)

train_set = targets = parsed_dataframe


semantic_types_to_remove = set(
    [
        "https://metadata.datadrivendiscovery.org/types/TrueTarget",
        "https://metadata.datadrivendiscovery.org/types/SuggestedTarget",
    ]
)
semantic_types_to_add = set(
    ["https://metadata.datadrivendiscovery.org/types/PredictedTarget"]
)

# We want to test the running of the code without errors and not the correctness of it
# since that is assumed to be tested by sklearn


class TestSKRidge(unittest.TestCase):
    def create_learner(self, hyperparams):
        clf = SKRidge.SKRidge(hyperparams=hyperparams)
        return clf

    def set_training_data_on_learner(self, learner, **args):
        learner.set_training_data(**args)

    def fit_learner(self, learner: PrimitiveBase):
        learner.fit()

    def produce_learner(self, learner, **args):
        return learner.produce(**args)

    def basic_fit(self, hyperparams):
        learner = self.create_learner(hyperparams)
        training_data_args = self.set_data(hyperparams)
        self.set_training_data_on_learner(learner, **training_data_args)

        self.assertRaises(
            PrimitiveNotFittedError,
            learner.produce,
            inputs=training_data_args.get("inputs"),
        )

        self.fit_learner(learner)

        assert len(learner._training_indices) > 0

        output = self.produce_learner(learner, inputs=training_data_args.get("inputs"))
        return output, learner, training_data_args

    def pickle(self, hyperparams):
        output, learner, training_data_args = self.basic_fit(hyperparams)

        # Testing get_params() and set_params()
        params = learner.get_params()
        learner.set_params(params=params)

        model = pickle.dumps(learner)
        new_clf = pickle.loads(model)
        new_output = new_clf.produce(inputs=training_data_args.get("inputs"))

        assert_frame_equal(new_output.value, output.value)

    def set_data(self, hyperparams):
        hyperparams = hyperparams.get("use_semantic_types")
        if hyperparams:
            return {"inputs": train_set, "outputs": targets}
        else:
            return {
                "inputs": parsed_dataframe.select_columns([1, 2, 3, 4]),
                "outputs": parsed_dataframe.select_columns([5]),
            }

    def get_transformed_indices(self, learner):
        return learner._target_column_indices

    def new_return_checker(self, output, indices):
        input_target = train_set.select_columns(list(indices))
        for i in range(len(output.columns)):
            input_semantic_types = input_target.metadata.query(
                (metadata_base.ALL_ELEMENTS, i)
            ).get("semantic_types")
            output_semantic_type = set(
                output.metadata.query((metadata_base.ALL_ELEMENTS, i)).get(
                    "semantic_types"
                )
            )
            transformed_input_semantic_types = (
                set(input_semantic_types) - semantic_types_to_remove
            )
            transformed_input_semantic_types = transformed_input_semantic_types.union(
                semantic_types_to_add
            )
            assert output_semantic_type == transformed_input_semantic_types

    def append_return_checker(self, output, indices):
        for i in range(len(train_set.columns)):
            input_semantic_types = set(
                train_set.metadata.query((metadata_base.ALL_ELEMENTS, i)).get(
                    "semantic_types"
                )
            )
            output_semantic_type = set(
                output.value.metadata.query((metadata_base.ALL_ELEMENTS, i)).get(
                    "semantic_types"
                )
            )
            assert output_semantic_type == input_semantic_types

        self.new_return_checker(
            output.value.select_columns(
                list(range(len(train_set.columns), len(output.value.columns)))
            ),
            indices,
        )

    def replace_return_checker(self, output, indices):
        for i in range(len(train_set.columns)):
            if i in indices:
                continue
            input_semantic_types = set(
                train_set.metadata.query((metadata_base.ALL_ELEMENTS, i)).get(
                    "semantic_types"
                )
            )
            output_semantic_type = set(
                output.value.metadata.query((metadata_base.ALL_ELEMENTS, i)).get(
                    "semantic_types"
                )
            )
            assert output_semantic_type == input_semantic_types

        self.new_return_checker(output.value.select_columns(list(indices)), indices)

    def test_with_semantic_types(self):
        hyperparams = SKRidge.Hyperparams.defaults().replace(
            {"use_semantic_types": True}
        )
        self.pickle(hyperparams)

    def test_without_semantic_types(self):
        hyperparams = SKRidge.Hyperparams.defaults()
        self.pickle(hyperparams)

    def test_with_new_return_result(self):
        hyperparams = SKRidge.Hyperparams.defaults().replace(
            {"return_result": "new", "use_semantic_types": True}
        )
        output, clf, _ = self.basic_fit(hyperparams)
        indices = self.get_transformed_indices(clf)
        self.new_return_checker(output.value, indices)

    def test_with_append_return_result(self):
        hyperparams = SKRidge.Hyperparams.defaults().replace(
            {"return_result": "append", "use_semantic_types": True}
        )
        output, clf, _ = self.basic_fit(hyperparams)
        indices = self.get_transformed_indices(clf)
        self.append_return_checker(output, indices)

    def test_with_replace_return_result(self):
        hyperparams = SKRidge.Hyperparams.defaults().replace(
            {"return_result": "replace", "use_semantic_types": True}
        )
        output, clf, _ = self.basic_fit(hyperparams)
        indices = self.get_transformed_indices(clf)
        self.replace_return_checker(output, indices)

    def test_produce_methods(self):
        hyperparams = SKRidge.Hyperparams.defaults()
        output, clf, _ = self.basic_fit(hyperparams)
        list_of_methods = [
            "produce_cluster_centers",
            "produce_feature_importances",
            "produce_support",
        ]
        for method in list_of_methods:
            produce_method = getattr(clf, method, None)
            if produce_method:
                produce_method()

    def test_target_column_name(self):
        hyperparams = SKRidge.Hyperparams.defaults().replace(
            {"return_result": "replace", "use_semantic_types": True}
        )
        output, clf, _ = self.basic_fit(hyperparams)

        predicted_target_column_list = (
            output.value.metadata.get_columns_with_semantic_type(
                "https://metadata.datadrivendiscovery.org/types/PredictedTarget"
            )
        )
        input_true_target_column_list = (
            parsed_dataframe.metadata.get_columns_with_semantic_type(
                "https://metadata.datadrivendiscovery.org/types/TrueTarget"
            )
        )
        # Test if metadata was copied correctly
        predicted_target_column_metadata = output.value.metadata.select_columns(
            predicted_target_column_list
        )
        input_true_target_column_metadata = parsed_dataframe.metadata.select_columns(
            input_true_target_column_list
        )

        if len(predicted_target_column_list) == 1:
            # Checking that the predicted target name matches the input target
            predicted_name = predicted_target_column_metadata.query(
                (metadata_base.ALL_ELEMENTS,)
            ).get("name")
            input_true_target_name = input_true_target_column_metadata.query(
                (metadata_base.ALL_ELEMENTS,)
            ).get("name")
            assert predicted_name == input_true_target_name


if __name__ == "__main__":
    unittest.main()
