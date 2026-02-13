"""Simulation-based imaging: numerical solvers and inverse models."""

from sbimaging.logging import configure_logging, get_logger

__version__ = "0.1.0"
__all__ = ["configure_logging", "get_logger"]


def list_presets() -> list[str]:
    """List available configuration presets.

    Returns:
        List of preset names.
    """
    from sbimaging.config import list_presets as _list_presets

    return _list_presets()


def load_preset(name: str):
    """Load a configuration preset by name.

    Args:
        name: Preset name (e.g., "simple_brain", "ellipsoid_sweep").

    Returns:
        ConfigPreset instance.
    """
    from sbimaging.config import load_preset as _load_preset

    return _load_preset(name)
