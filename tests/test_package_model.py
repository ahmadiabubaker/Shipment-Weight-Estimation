"""Regression test for the packaging gap: pyproject.toml declares
shipment_weight/models/*.joblib as package data, but scripts/train_real_data.py
saves to the top-level models/ dir, and nothing copied the artifact into the
package tree before this -- a wheel built with `python -m build` would not
have actually contained the model, and ShipmentWeightPredictor.load() would
fail once installed outside a repo checkout (no repo-relative fallback to
land on).

test_package_model_copies_bundle_into_src_tree is fast and always runs.
test_built_wheel_contains_model_artifact is the real regression check -- it
builds an actual wheel and inspects it as a zip archive, because a
package-data glob matching a path is not the same guarantee as "the file is
really inside the wheel" (e.g. the src tree could be empty at build time).
It's slower and needs `build`/`wheel`/`setuptools>=68` installed (part of
the `dev` extra), so it skips cleanly if that tooling isn't present rather
than failing an otherwise-unrelated `pip install -e .` setup.

Both tests save/restore src/shipment_weight/models/model.joblib around
themselves -- that file is a real, meaningful local artifact (not
something generated fresh per test run), so clobbering it on a failed
assertion would be a bad side effect of running the test suite.
"""
from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import joblib
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_MODEL_PATH = REPO_ROOT / "src" / "shipment_weight" / "models" / "model.joblib"
PACKAGE_SCRIPT = REPO_ROOT / "scripts" / "package_model.py"


@pytest.fixture
def preserved_packaged_model():
    """Saves whatever is currently at PACKAGE_MODEL_PATH (or notes its
    absence) and restores that exact state after the test, pass or fail."""
    original = PACKAGE_MODEL_PATH.read_bytes() if PACKAGE_MODEL_PATH.is_file() else None
    try:
        yield
    finally:
        if original is not None:
            PACKAGE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            PACKAGE_MODEL_PATH.write_bytes(original)
        elif PACKAGE_MODEL_PATH.is_file():
            PACKAGE_MODEL_PATH.unlink()


def _fake_bundle(tmp_path: Path) -> Path:
    bundle_path = tmp_path / "fake_model.joblib"
    joblib.dump({"pipeline": None, "model_type": "fake", "model_version": "test"}, bundle_path)
    return bundle_path


def test_package_model_copies_bundle_into_src_tree(tmp_path, preserved_packaged_model):
    fake = _fake_bundle(tmp_path)

    result = subprocess.run(
        [sys.executable, str(PACKAGE_SCRIPT), "--model", str(fake)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    assert PACKAGE_MODEL_PATH.is_file(), "package_model.py did not create src/shipment_weight/models/model.joblib"
    copied = joblib.load(PACKAGE_MODEL_PATH)
    assert copied["model_type"] == "fake"  # confirms it copied OUR fake bundle, not a stale leftover


def test_package_model_errors_clearly_on_missing_source(tmp_path, preserved_packaged_model):
    missing = tmp_path / "does_not_exist.joblib"
    result = subprocess.run(
        [sys.executable, str(PACKAGE_SCRIPT), "--model", str(missing)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "No model artifact" in result.stderr


@pytest.mark.slow
def test_built_wheel_contains_model_artifact(tmp_path, preserved_packaged_model):
    """The real regression check for the original gap: build an actual
    wheel and confirm shipment_weight/models/model.joblib is inside it."""
    pytest.importorskip("build")

    fake = _fake_bundle(tmp_path)
    subprocess.run(
        [sys.executable, str(PACKAGE_SCRIPT), "--model", str(fake)],
        check=True, capture_output=True, text=True,
    )

    dist_dir = tmp_path / "dist"
    build = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(dist_dir), str(REPO_ROOT)],
        capture_output=True, text=True,
    )
    assert build.returncode == 0, (
        "python -m build failed -- if this is `setuptools`/`wheel` missing/outdated, "
        f"they're part of the `dev` extra (pip install -e '.[dev]').\n{build.stdout}\n{build.stderr}"
    )

    wheels = list(dist_dir.glob("*.whl"))
    assert wheels, f"no wheel produced in {dist_dir}"

    with zipfile.ZipFile(wheels[0]) as zf:
        names = zf.namelist()
    assert any(n.endswith("shipment_weight/models/model.joblib") for n in names), (
        f"built wheel does not contain the model artifact -- package-data declaration in "
        f"pyproject.toml and the actual wheel contents have drifted apart. Wheel contents: {names}"
    )
