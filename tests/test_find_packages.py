import os

import pytest
from packaging.requirements import Requirement

from launch.find_packages import (
    EPP_NO_ERROR,
    EPP_PKG_NOT_EXIST,
    EPP_PKG_VERSION_MISMATCH,
    ModuleManager,
)


class FakeDistribution:
    """Stands in for an importlib.metadata.Distribution."""

    def __init__(self, name, version, location, top_level=None):
        self.metadata = {"Name": name}
        self.version = version
        self._location = location
        self._top_level = top_level

    def locate_file(self, path):
        return os.path.join(self._location, path)

    def read_text(self, filename):
        return self._top_level if filename == "top_level.txt" else None


@pytest.fixture
def index_distributions(mocker):
    """Build a ModuleManager over a fixed set of distributions."""

    def _index(dists):
        mocker.patch("importlib.metadata.distributions", return_value=iter(dists))
        return ModuleManager()

    return _index


def test_distribution_names_are_canonicalized(index_distributions, tmp_path):
    # importlib.metadata reports the raw Name field, so "Typing_Extensions" and
    # "jaraco.text" have to be normalized to the names requirements refer to.
    manager = index_distributions(
        [
            FakeDistribution("Typing_Extensions", "4.16.0", str(tmp_path), "typing_extensions\n"),
            FakeDistribution("jaraco.text", "4.0.0", str(tmp_path), "jaraco\n"),
        ]
    )

    assert manager.pip_pkg_map == {"typing-extensions": "4.16.0", "jaraco-text": "4.0.0"}
    assert manager.pip_module_map["typing_extensions"] == [("typing-extensions", "4.16.0")]


def test_verify_pkg_matches_alternate_name_spellings(index_distributions, tmp_path):
    manager = index_distributions(
        [FakeDistribution("typing_extensions", "4.16.0", str(tmp_path), "typing_extensions\n")]
    )

    assert manager.verify_pkg(Requirement("typing-extensions>=4.0")) == EPP_NO_ERROR
    assert manager.verify_pkg(Requirement("Typing_Extensions>=4.0")) == EPP_NO_ERROR
    assert manager.verify_pkg(Requirement("typing.extensions>=4.0")) == EPP_NO_ERROR
    assert manager.verify_pkg(Requirement("typing-extensions<4.0")) == EPP_PKG_VERSION_MISMATCH
    assert manager.verify_pkg(Requirement("not-installed>=1.0")) == EPP_PKG_NOT_EXIST


def test_first_distribution_on_sys_path_wins(index_distributions, tmp_path):
    # distributions() also yields shadowed copies, e.g. a vendored tree later on
    # sys.path. The earlier one is the one that actually gets imported.
    installed = tmp_path / "site-packages"
    installed.mkdir()
    vendored = tmp_path / "vendored"
    vendored.mkdir()

    manager = index_distributions(
        [
            FakeDistribution("packaging", "26.3", str(installed), "packaging\n"),
            FakeDistribution("packaging", "26.0", str(vendored), "packaging\n"),
        ]
    )

    assert manager.pip_pkg_map["packaging"] == "26.3"
    assert manager.pip_module_map["packaging"] == [("packaging", "26.3")]
    assert str(vendored) not in manager.nonlocal_package_path


def test_distribution_paths_are_realpath_normalized(index_distributions, tmp_path):
    # is_local_path() compares these paths by identity, so an unresolved symlink
    # would stop a nonlocal package from being recognized as one.
    installed = tmp_path / "real-site-packages"
    installed.mkdir()
    symlinked = tmp_path / "linked-site-packages"
    symlinked.symlink_to(installed)

    manager = index_distributions([FakeDistribution("some-pkg", "1.0.0", str(symlinked), "some_pkg\n")])

    assert os.path.realpath(str(installed)) in manager.nonlocal_package_path
    assert str(symlinked) not in manager.nonlocal_package_path


def test_distributions_without_a_name_are_skipped(index_distributions, tmp_path):
    manager = index_distributions(
        [
            FakeDistribution(None, "1.0.0", str(tmp_path), "broken\n"),
            FakeDistribution("good-pkg", "2.0.0", str(tmp_path), "good_pkg\n"),
        ]
    )

    assert manager.pip_pkg_map == {"good-pkg": "2.0.0"}
    assert "broken" not in manager.pip_module_map


def test_missing_or_blank_top_level_metadata_is_tolerated(index_distributions, tmp_path):
    # Wheels are not required to ship top_level.txt, and the ones that do may end
    # with a trailing newline.
    manager = index_distributions(
        [
            FakeDistribution("no-top-level", "1.0.0", str(tmp_path), None),
            FakeDistribution("blank-lines", "2.0.0", str(tmp_path), "blank_lines\n\n"),
        ]
    )

    assert manager.pip_pkg_map == {"no-top-level": "1.0.0", "blank-lines": "2.0.0"}
    assert manager.pip_module_map == {"blank_lines": [("blank-lines", "2.0.0")]}


def test_setuptools_modules_are_tracked_separately(index_distributions, tmp_path):
    manager = index_distributions(
        [
            FakeDistribution("setuptools", "80.10.2", str(tmp_path), "setuptools\npkg_resources\n"),
            FakeDistribution("requests", "2.32.0", str(tmp_path), "requests\n"),
        ]
    )

    assert manager.setuptools_module_set == {"setuptools", "pkg_resources"}
    assert manager.pip_module_map == {"requests": [("requests", "2.32.0")]}
