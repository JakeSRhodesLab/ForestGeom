from pathlib import Path


def test_proximity_forests_is_an_optional_backend():
    source = Path("src/forestgeom/adapters/__init__.py").read_text(encoding="utf-8")
    assert "from proximity_forests import" in source
    assert "except ImportError:" in source
    assert "_PROXIMITY_FOREST_CLASSES = ()" in source
