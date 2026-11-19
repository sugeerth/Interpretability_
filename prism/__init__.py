"""Prism: agentic interpretability for open-weights and API-only language models."""
from .backend import Hook, WhiteBoxModel
from .workspace import Workspace

__all__ = ["Hook", "WhiteBoxModel", "Workspace"]
__version__ = "0.1.0"
