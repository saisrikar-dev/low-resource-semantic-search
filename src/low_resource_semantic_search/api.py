from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from .pipeline import SemanticSearchEngine


class _BaseSearchHandler(BaseHTTPRequestHandler):
    engine: SemanticSearchEngine

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=True, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/health":
            self._send_json(404, {"error": "Not found"})
            return
        self._send_json(200, {"status": "ok"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/search":
            self._send_json(404, {"error": "Not found"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        payload = self.rfile.read(content_length)

        try:
            body = json.loads(payload or b"{}")
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Body must be valid JSON"})
            return

        query = str(body.get("query", "")).strip()
        top_k = int(body.get("top_k", 5))
        if not query:
            self._send_json(400, {"error": "The 'query' field is required"})
            return

        response = self.engine.search(query, top_k=top_k)
        self._send_json(200, response.to_dict())


def run_api_server(engine: SemanticSearchEngine, host: str, port: int) -> None:
    handler = type(
        "SearchHandler",
        (_BaseSearchHandler,),
        {"engine": engine},
    )
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Serving search API on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

