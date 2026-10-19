"""Optional display entry point. Importing data_audit does not import this module."""


def show(result, *, port=None, no_browser=False):
    """Open existing results without rerunning analysis. Blocks until the UI stops."""
    import os
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from ..cli import serve

    with TemporaryDirectory(prefix="data-audit-view-") as directory:
        path = Path(directory) / "result.audit.zip"
        result.save(path)
        environ = os.environ.copy()
        environ["DATA_AUDIT_RESULT"] = str(path)
        return serve(port=port, no_browser=no_browser, environ=environ)
