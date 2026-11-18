"""One-command launcher for SindhiVoice AI Desktop Application."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from server import run_server


def _launch_desktop_window(url: str) -> None:
    time.sleep(0.8)
    # 1. Try launching in standalone desktop app window mode via Chromium/Chrome
    chrome_bins = ["chromium", "google-chrome", "google-chrome-stable", "chrome"]
    for b in chrome_bins:
        bin_path = shutil.which(b)
        if bin_path:
            try:
                subprocess.Popen(
                    [
                        bin_path,
                        f"--app={url}",
                        "--window-size=1200,820",
                        "--disable-sync",
                        "--no-first-run",
                        "--use-fake-ui-for-media-stream",  # Auto-grant mic in app mode
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return
            except Exception:
                pass

    # 2. Fallback to default browser
    try:
        webbrowser.open(url)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SindhiVoice AI — Voice-Based Sindhi to English Desktop App"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8501, help="Port number (default: 8501)")
    parser.add_argument("--no-app", action="store_true", help="Do not auto-open desktop window")
    parser.add_argument("--tk", action="store_true", help="Launch native Tkinter desktop window")
    args = parser.parse_args()

    if args.tk:
        try:
            import desktop_app
            desktop_app.main()
            return
        except Exception as e:
            print(f"[Notice] Tkinter desktop launch fallback ({e}); launching desktop app window...")

    url = f"http://{args.host}:{args.port}"
    print("=" * 68)
    print("  SindhiVoice AI  —  Voice-Based Sindhi to English Translator")
    print("  'NLP: How It Works? — Class 9 & 10 AI Project'")
    print("=" * 68)
    print(f"  • Desktop Window URL: {url}")
    print("  • Voice Input       : Live Microphone (Speak Sindhi)")
    print("  • Voice Output      : English Voice Speech Aloud (TTS)")
    print("  • Press Ctrl+C to close the application.")
    print("=" * 68)

    if not args.no_app:
        threading.Thread(target=_launch_desktop_window, args=(url,), daemon=True).start()

    server = run_server(host=args.host, port=args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nSindhiVoice AI closed safely. Goodbye!")
        server.server_close()


if __name__ == "__main__":
    main()
