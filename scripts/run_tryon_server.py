"""Run the PostVTON try-on FastAPI server.

Usage:
  python scripts/run_tryon_server.py --host 0.0.0.0 --port 8000

Environment:
  TRYON_SERVER_DEVICE=cuda|cpu
  TRYON_SERVER_OUTPUT_DIR=outputs

This script is just a thin wrapper around uvicorn.
"""

from __future__ import annotations

import argparse


def main() -> int:
	parser = argparse.ArgumentParser(description="Run PostVTON try-on server")
	parser.add_argument("--host", default="0.0.0.0")
	parser.add_argument("--port", type=int, default=8000)
	args = parser.parse_args()

	try:
		import uvicorn
	except Exception as exc:  # pragma: no cover
		raise SystemExit(
			"uvicorn is required. Install server deps with: pip install -r requirements.tryon_server.txt"
		) from exc

	uvicorn.run(
		"postvton.tryon_server.app:app",
		host=args.host,
		port=args.port,
		reload=False,
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
