# D3M Estimator to Primitive

[![.github/workflows/ci.yml](https://github.com/thomasjpfan/d3m_estimator_to_primitive/actions/workflows/ci.yml/badge.svg)](https://github.com/thomasjpfan/d3m_estimator_to_primitive/actions/workflows/ci.yml)

Converts sklearn compatible estimators into D3M primitives

## Installation

0. Create an virtual environment.
1. Install d3m_estimator_to_primitive

```bash
pip install git+https://github.com/thomasjpfan/d3m_estimator_to_primitive
```

2. Install library that contains sklearn compatible estimator. For example, if you want to wrap `xgboost.XGBClassifier`, install `xgboost`.
3. Create a `metadata.yml` for estimators you want to convert. See the `integration_test` folder for examples.
4. Run the conversion script:

```bash
estimator_to_primitive integration_test/xgboost_metadata.yml
```

This generates code for a D3M primitives using the package name provided in `xgboost_metadata.yaml`:

```
xgboost_wrap
├── D3M_XGBClassifier.py
├── tests
│   └── test_D3M_XGBClassifier.py
├── tests-data
│   └── datasets
│       └── iris_dataset_1
│           ├── datasetDoc.json
│           └── tables
│               └── learningData.csv
└── vendor
    ├── __init__.py
    └── common_primitives
        ├── __init__.py
        ├── __pycache__
        ├── column_parser.py
        └── dataset_to_dataframe.py
```

The `test-data` and `vendor` directory are used for running in the `tests` directory. To run the tests:

```bash
PYTHONPATH=$PWD python xgboost_wrap/tests/test_D3M_XGBClassifier.py
```

## License

This repo is under the [MIT License](LICENSE).
