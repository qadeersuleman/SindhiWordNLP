"""Lightweight, pure-Python server for SindhiVoice AI.

Provides the REST API for Sindhi NLP prediction, voice TTS output, and serves the
desktop app interface without requiring any external pip dependencies.
"""

from __future__ import annotations

import json
import mimetypes
import shutil
import subprocess
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

# Import Sindhi NLP Model
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
from sindhi_nlp_demo import SINDHI_METADATA, SindhiWordClassifier

STATIC_DIR = BASE_DIR / "static"
classifier = SindhiWordClassifier()


def speak_local_tts(text: str) -> None:
    """Trigger system TTS on Linux (spd-say / espeak) in a background thread."""
    def _worker():
        clean_text = text.replace('"', "").replace("'", "").strip()
        if not clean_text:
            return
        if shutil.which("spd-say"):
            try:
                subprocess.run(
                    ["spd-say", "-o", "espeak", "-r", "-10", clean_text],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return
            except Exception:
                pass
        if shutil.which("espeak"):
            try:
                subprocess.run(
                    ["espeak", "-s", "145", clean_text],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass

    threading.Thread(target=_worker, daemon=True).start()


class SindhiVoiceHTTPHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        # Suppress routine GET logging for cleaner terminal output
        if "GET /static" not in self.requestline:
            super().log_message(format, *args)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self._serve_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            return

        if path.startswith("/static/"):
            rel_path = path[len("/static/") :]
            target = STATIC_DIR / rel_path
            mime, _ = mimetypes.guess_type(str(target))
            self._serve_file(target, mime or "application/octet-stream")
            return

        if path.startswith("/styles.css") or path.startswith("/app.js"):
            target = STATIC_DIR / path.lstrip("/")
            mime, _ = mimetypes.guess_type(str(target))
            self._serve_file(target, mime or "application/octet-stream")
            return

        if path == "/api/predict":
            query_params = parse_qs(parsed.query)
            q = query_params.get("q", [""])[0]
            if not q:
                self._send_json({"error": "No query provided"}, status=400)
                return

            q = unquote(q)
            pred_class, conf, all_probs = classifier.predict(q)
            meta = SINDHI_METADATA.get(pred_class, {})

            payload = {
                "ok": True,
                "input": q,
                "class": pred_class,
                "confidence": conf,
                "sindhi_primary": meta.get("sindhi_primary", q),
                "phonetic": meta.get("phonetic", ""),
                "english": meta.get("meaning", ""),
                "spoken_text": meta.get("meaning", ""),
                "desc": meta.get("desc", ""),
                "probabilities": all_probs,
            }
            self._send_json(payload)
            return

        if path == "/api/speak":
            query_params = parse_qs(parsed.query)
            text = query_params.get("text", [""])[0]
            if text:
                speak_local_tts(unquote(text))
            self._send_json({"ok": True, "spoken": text})
            return

        self.send_error(HTTPStatus.NOT_FOUND, "File not found")

    def _serve_file(self, file_path: Path, content_type: str) -> None:
        if not file_path.exists() or not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return

        data = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, data: dict, status: int = 200) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(raw)


def run_server(host: str = "127.0.0.1", port: int = 8501) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), SindhiVoiceHTTPHandler)
    return server


if __name__ == "__main__":
    srv = run_server()
    print("Serving on http://127.0.0.1:8501")
    srv.serve_forever()
