#!/usr/bin/env python3
"""
Start the program by running python3 run_program.py in the terminal
Ctrl+C will stop the program
"""

import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
FRONTEND_DIR = PROJECT_ROOT / "frontend"


def stream_output(prefix: str, pipe):
    try:
        for line in iter(pipe.readline, ""):
            if not line:
                break
            sys.stdout.write(f"[{prefix}] {line}")
            sys.stdout.flush()
    finally:
        pipe.close()


def start_backend():
    print("Starting backend: python3 app.py")
    return subprocess.Popen(
        [sys.executable, "app.py"],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )


def start_frontend():
    print("Starting frontend: cd frontend && npm start")
    env = os.environ.copy()
    env.setdefault("CI", "false")
    env.setdefault("BROWSER", "none")
    return subprocess.Popen(
        ["npm", "start"],
        cwd=str(FRONTEND_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
        env=env,
    )


def ensure_frontend_dependencies():
    bin_path = FRONTEND_DIR / "node_modules" / ".bin" / "react-scripts"
    if bin_path.exists():
        return True
    print("react-scripts not found; running npm install in frontend/ ...")
    try:
        subprocess.check_call(["npm", "install"], cwd=str(FRONTEND_DIR))
    except subprocess.CalledProcessError as exc:
        print(f"npm install failed (exit {exc.returncode}). Fix and retry.")
        return False
    return bin_path.exists()


def stop_process(proc, name: str):
    if not proc:
        return
    if proc.poll() is not None:
        return
    print(f"Stopping {name} (pid {proc.pid})")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


def main():
    if not FRONTEND_DIR.exists():
        print("frontend/ directory not found.")
        sys.exit(1)

    if not ensure_frontend_dependencies():
        sys.exit(1)

    backend = start_backend()
    frontend = start_frontend()

    threads = [
        threading.Thread(target=stream_output, args=("backend", backend.stdout), daemon=True),
        threading.Thread(target=stream_output, args=("frontend", frontend.stdout), daemon=True),
    ]
    for t in threads:
        t.start()

    def open_browser_later():
        url = "http://localhost:3000"
        deadline = time.time() + 30  # wait up to 30s for dev server

        # wait for the dev server to start
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=1):
                    break
            except (urllib.error.URLError, TimeoutError, ConnectionResetError):
                time.sleep(1)
        else:
            # give a tiny grace period even if we never saw it come up
            time.sleep(2)

        try:
            webbrowser.open(url)
            print(f"Opening browser at {url}")
        except Exception as exc:
            print(f"Could not open browser automatically: {exc}")

    threading.Thread(target=open_browser_later, daemon=True).start()

    print("Servers running. Backend: http://localhost:5001  Frontend: http://localhost:3000")
    print("Press Ctrl+C to stop both.")

    try:
        while True:
            backend_ret = backend.poll()
            frontend_ret = frontend.poll()
            if backend_ret is not None:
                print(f"Backend exited with code {backend_ret}")
                break
            if frontend_ret is not None:
                print(f"Frontend exited with code {frontend_ret}")
                break
            # simple wait loop
            try:
                backend.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
    except KeyboardInterrupt:
        print("\nReceived Ctrl+C, stopping servers...")
    finally:
        stop_process(frontend, "frontend")
        stop_process(backend, "backend")
        print("All servers stopped.")


if __name__ == "__main__":
    main()
