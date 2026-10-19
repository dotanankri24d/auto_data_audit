"""Data Audit: inspect label issues and overlap without importing a UI or fitting on import."""
from .api import audit, bind_embedding, bind_distance, bind_method, bind_policy
from .contracts import AnalysisInput, CancellationToken, ValidationError
from .embeddings import Identity, Standardize, PCAEmbedding, EmbeddingPipeline, AutoencoderEmbedding
from .distances import Euclidean
from .methods import NeighborAudit, OOFAudit
from .policies import ReviewPolicy
from .io import load_csv, load_excel, list_excel_sheets, map_columns
from .result import AuditResult, DecisionResult

__version__ = "0.1.0"


def load_run(source):
    from .io.bundle import load_run as load
    return load(source)


__all__ = ["audit", "bind_embedding", "bind_distance", "bind_method", "bind_policy", "AnalysisInput",
           "CancellationToken", "ValidationError", "Identity", "Standardize", "PCAEmbedding", "EmbeddingPipeline",
           "AutoencoderEmbedding", "Euclidean", "NeighborAudit", "OOFAudit", "ReviewPolicy", "load_csv",
           "load_excel", "list_excel_sheets", "map_columns", "AuditResult", "DecisionResult", "load_run"]
