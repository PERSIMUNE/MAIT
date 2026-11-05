"""Access to the original MAIT pipeline without importing analysis dependencies."""
from .export import export_demo

__version__ = "0.1.0a4"
__all__ = ["export_demo", "__version__"]
