"""Serve the standalone frontend and proxy API requests to the existing FastAPI app."""

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


FRONTEND_DIR = Path(__file__).resolve().parent
BACKEND_URL = "http://127.0.0.1:8000"


class FrontendHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def _proxy(self):
        target = f"{BACKEND_URL}{self.path[4:]}"
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else None
        request = Request(
            target,
            data=body,
            method=self.command,
            headers={"Content-Type": self.headers.get("Content-Type", "application/json")},
        )

        try:
            with urlopen(request, timeout=30) as response:
                payload = response.read()
                self.send_response(response.status)
                self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
        except HTTPError as error:
            payload = error.read()
            self.send_response(error.code)
            self.send_header("Content-Type", error.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except URLError:
            self.send_error(502, "FastAPI backend is unavailable")

    def do_GET(self):
        if self.path.startswith("/api/"):
            self._proxy()
        else:
            super().do_GET()

    def do_POST(self):
        if self.path.startswith("/api/"):
            self._proxy()
        else:
            self.send_error(405, "Method not allowed")


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 5500), FrontendHandler)
    print("Frontend available at http://127.0.0.1:5500")
    print("Proxying /api/* to http://127.0.0.1:8000/*")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping frontend server.")
    finally:
        server.server_close()
