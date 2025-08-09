from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import urlparse

from .paths import ensure_dirs, PLOTS_DIR, TRADES_DIR, STATS_DIR
from . import pages
from . import api


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class WebHandler(BaseHTTPRequestHandler):
    server_version = "TradeWebUI/0.2"

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            return self._send_bytes(pages.backtest_index(), content_type="text/html; charset=utf-8")
        if path == "/chart":
            return self._send_bytes(pages.chart_page(), content_type="text/html; charset=utf-8")
        if path == "/echarts":
            return self._send_bytes(pages.echarts_page(), content_type="text/html; charset=utf-8")
        if path == "/echarts-mtf":
            return self._send_bytes(pages.echarts_mtf_page(), content_type="text/html; charset=utf-8")
        if path == "/api/ohlcv":
            return api.api_ohlcv(self, parsed)
        if path == "/api/mtf":
            return api.api_mtf(self, parsed)
        if path.startswith("/static/"):
            return self._serve_static(path)
        return self._send_text("Not Found", 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/backtest":
            return api.handle_backtest(self)
        if parsed.path == "/chart":
            # Render server-side static chart and respond with image page
            # Keep backward-compatible by delegating to /backtest HTML? For now, show form again.
            return self._send_bytes(pages.chart_page(), content_type="text/html; charset=utf-8")
        return self._send_text("Not Found", 404)

    def _serve_static(self, path: str):
        # /static/{kind}/{file}
        import os
        parts = path.split("/")
        if len(parts) < 4:
            return self._send_text("Bad request", 400)
        kind = parts[2]
        name = "/".join(parts[3:])
        safe_name = os.path.normpath(name)
        if safe_name.startswith(".."):
            return self._send_text("Forbidden", 403)
        base = {"plots": PLOTS_DIR, "trades": TRADES_DIR, "stats": STATS_DIR}.get(kind)
        if not base:
            return self._send_text("Not Found", 404)
        fpath = os.path.join(base, safe_name)
        if not os.path.exists(fpath):
            return self._send_text("Not Found", 404)
        try:
            with open(fpath, "rb") as f:
                data = f.read()
            ctype = "application/octet-stream"
            if fpath.endswith(".png"):
                ctype = "image/png"
            elif fpath.endswith(".json"):
                ctype = "application/json"
            elif fpath.endswith(".csv"):
                ctype = "text/csv"
            return self._send_bytes(data, content_type=ctype)
        except Exception as e:
            return self._send_text(f"Error: {e}", 500)

    def _send_text(self, text: str, code: int = 200):
        data = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_bytes(self, data: bytes, *, content_type: str = "application/octet-stream", code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main(argv: list[str] | None = None):
    import argparse
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Trade Platform WebUI (modular)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    srv = ThreadingHTTPServer((args.host, args.port), WebHandler)
    print(f"WebUI running: http://{args.host}:{args.port}")
    try:
        srv.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        srv.server_close()

