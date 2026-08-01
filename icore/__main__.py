"""
icore CLI entry point.

Run the icore API server:

    python -m icore                      # default host:port from config
    python -m icore --host 0.0.0.0 --port 8000 --reload
    icore                                # installed console script

The server uses the production bootstrap (``icore.bootstrap``), which
loads ``config/*.yaml``, builds the Model/DB managers, and auto-registers
example workflows before serving.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    """Parse CLI args and launch uvicorn with the production app."""
    parser = argparse.ArgumentParser(
        prog="icore",
        description="icore - Enterprise LLM Workflow Orchestration Platform",
    )
    parser.add_argument("--host", default=None, help="Bind host (default: from config)")
    parser.add_argument("--port", type=int, default=None, help="Bind port (default: from config)")
    parser.add_argument("--workers", type=int, default=1, help="Uvicorn worker count")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev)")
    parser.add_argument(
        "--log-level",
        default=None,
        help="Log level (debug/info/warning/error)",
    )
    args = parser.parse_args(argv)

    # Resolve defaults from icore settings (env vars / .env)
    from icore.config import get_settings

    settings = get_settings()
    host = args.host or settings.api.host
    port = args.port or settings.api.port
    log_level = args.log_level or settings.logging.level.lower()

    try:
        import uvicorn
    except ImportError:
        print(
            "uvicorn is not installed. Install with: pip install uvicorn[standard]",
            file=sys.stderr,
        )
        return 1

    print(
        f"Starting icore v{settings.version} on http://{host}:{port} "
        f"(workers={args.workers}, reload={args.reload})",
        file=sys.stderr,
    )

    uvicorn.run(
        "icore.api.main:app",
        host=host,
        port=port,
        workers=args.workers if not args.reload else 1,
        reload=args.reload,
        log_level=log_level,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
