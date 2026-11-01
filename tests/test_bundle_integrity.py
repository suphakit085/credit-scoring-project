import joblib
import pytest

from src.inference import REQUIRED_BUNDLE_KEYS, digest_path, load_bundle, write_digest


@pytest.fixture
def bundle_file(tmp_path):
    path = tmp_path / "bundle.joblib"
    joblib.dump({key: 0 for key in REQUIRED_BUNDLE_KEYS}, path)
    return path


def test_bundle_with_matching_digest_loads(bundle_file):
    write_digest(bundle_file)
    assert set(load_bundle(bundle_file)) == REQUIRED_BUNDLE_KEYS


def test_tampered_bundle_is_refused_before_unpickling(bundle_file):
    write_digest(bundle_file)
    with open(bundle_file, "ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="does not match its recorded SHA-256"):
        load_bundle(bundle_file)


def test_bundle_without_digest_loads_with_a_warning(bundle_file):
    assert not digest_path(bundle_file).exists()
    with pytest.warns(UserWarning, match="without an integrity check"):
        load_bundle(bundle_file)


def test_digest_file_uses_sha256sum_format(bundle_file):
    """Compared as bytes: read_text would hide a Windows "\\r\\n" that breaks sha256sum -c."""
    digest = write_digest(bundle_file)
    expected = f"{digest}  {bundle_file.name}\n".encode("utf-8")
    assert digest_path(bundle_file).read_bytes() == expected
