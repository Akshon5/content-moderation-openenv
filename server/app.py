"""
server/app.py — server entry point for OpenEnv multi-mode deployment.
Imports the FastAPI app from the root app.py and exposes a main() function
as required by the [project.scripts] entry point specification.
"""

import os
import sys

# Ensure root directory is on path so root app.py is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # noqa: F401 — re-export for openenv


def main():
    """server entry point called by: uv run server"""
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()