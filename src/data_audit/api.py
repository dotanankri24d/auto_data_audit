"""Public computational API; imports no UI dependencies."""

from .runner import audit
from .plugins import bind_embedding, bind_distance, bind_method, bind_policy

__all__ = ["audit", "bind_embedding", "bind_distance", "bind_method", "bind_policy"]
