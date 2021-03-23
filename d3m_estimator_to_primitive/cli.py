from pathlib import Path
import argparse
from shutil import copytree, rmtree, ignore_patterns
import uuid

from jinja2 import Environment, PackageLoader
from black import format_str, FileMode
from ruamel.yaml import YAML
from autoflake import fix_code


from ._utils import _get_estimator_cls, _train_estimator, _extract_metadata
from ._template import _prepare_for_templating


def _clean_and_write(path, source):
    remove_unused_imports = fix_code(source, remove_all_unused_imports=True)
    formated_python = format_str(remove_unused_imports, mode=FileMode())
    with path.open("w") as f:
        print(f"writing to {path}")
        f.write(formated_python)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_file", type=argparse.FileType("r"))

    args = parser.parse_args()

    yaml = YAML()
    metadata = yaml.load(args.metadata_file)

    version = metadata["version"]
    package_name = metadata["package_name"]
    package_dir = Path(package_name)
    package_dir.mkdir(exist_ok=True)

    data_dir = Path(__file__).parent / "data"
    src_folders = ["tests-data", "vendor"]

    for src in src_folders:
        target_dir = package_dir / src
        if target_dir.exists():
            rmtree(target_dir, ignore_errors=True)
        copytree(data_dir / src, package_dir / src, ignore=ignore_patterns("*.pyc"))

    # move files to testing
    test_dir = package_dir / "tests"
    test_dir.mkdir(exist_ok=True)

    estimators_meta = metadata["estimators"]

    e = Environment(
        loader=PackageLoader("d3m_estimator_to_primitive", "templates"),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    for estimator_name, user_metadata in estimators_meta.items():
        class_name = user_metadata["class_name"]
        Estimator = _get_estimator_cls(class_name)
        est = _train_estimator(Estimator)

        trained_metadata = _extract_metadata(est)

        template_metadata = _prepare_for_templating(
            estimator_name, user_metadata, trained_metadata
        )

        template = e.get_template("supervised.jinja2")
        unique_name = f"{class_name}{version}"
        rendered = template.render(
            **template_metadata,
            author=metadata["author"],
            version=version,
            id=str(uuid.uuid3(uuid.NAMESPACE_DNS, unique_name)),
            primitive_family=trained_metadata["primitive_family"],
        )

        _clean_and_write(package_dir / f"{estimator_name}.py", rendered)

        # generate tests
        test_template = e.get_template("test_supervised.jinja2")
        rendered = test_template.render(
            **template_metadata,
            package_name=package_name,
        )
        _clean_and_write(test_dir / f"test_{estimator_name}.py", rendered)
