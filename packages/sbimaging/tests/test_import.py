"""Tests for package imports."""


def test_import_sbimaging():
    """sbimaging package should be importable."""
    import sbimaging

    assert hasattr(sbimaging, "__version__")


def test_import_simulators():
    """Simulator modules should be importable."""
    from sbimaging import simulators
    from sbimaging.simulators import dg, fdtd

    assert simulators is not None
    assert fdtd is not None
    assert dg is not None


def test_import_inverse_models():
    """Inverse model modules should be importable."""
    from sbimaging import inverse_models
    from sbimaging.inverse_models import gp, nn

    assert inverse_models is not None
    assert gp is not None
    assert nn is not None


def test_import_array():
    """Array backend should be importable."""
    from sbimaging.array import xp

    assert xp is not None
