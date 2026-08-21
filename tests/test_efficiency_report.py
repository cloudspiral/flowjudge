import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).parents[1] / "scripts" / "build_efficiency_report.py"
    spec = importlib.util.spec_from_file_location("build_efficiency_report", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_efficiency_formatters_are_assignment_readable() -> None:
    module = _load_module()

    assert module._percent(0.625) == "62.5%"
    assert module._score(3.25) == "3.25/4"
