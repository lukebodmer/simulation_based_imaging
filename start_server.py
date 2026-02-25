#!/usr/bin/env python3
"""Start all servers for the Simulation Based Imaging platform.

Launches Django backend, React frontend dev server, and optionally
Trame control panel apps in separate processes.
"""

import argparse
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ServerProcess:
    """Represents a managed server process."""

    name: str
    process: subprocess.Popen


class ServerManager:
    """Manages multiple server processes."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.processes: list[ServerProcess] = []
        self._shutting_down = False

    def start_django(self, port: int = 8000) -> None:
        """Start the Django development server."""
        backend_dir = self.project_root / "backend"
        if not backend_dir.exists():
            print(f"Backend directory not found: {backend_dir}")
            return

        manage_py = backend_dir / "manage.py"
        if not manage_py.exists():
            print(f"Django manage.py not found: {manage_py}")
            return

        process = subprocess.Popen(
            [sys.executable, "manage.py", "runserver", f"0.0.0.0:{port}"],
            cwd=backend_dir,
        )
        self.processes.append(ServerProcess(name="django", process=process))
        print(f"Django server started on http://localhost:{port}")

    def start_frontend(self, port: int = 5173) -> None:
        """Start the Vite development server."""
        frontend_dir = self.project_root / "frontend"
        if not frontend_dir.exists():
            print(f"Frontend directory not found: {frontend_dir}")
            return

        package_json = frontend_dir / "package.json"
        if not package_json.exists():
            print(f"Frontend package.json not found: {package_json}")
            return

        process = subprocess.Popen(
            ["npm", "run", "dev", "--", "--port", str(port)],
            cwd=frontend_dir,
        )
        self.processes.append(ServerProcess(name="frontend", process=process))
        print(f"Frontend dev server started on http://localhost:{port}")

    def start_trame(self, app_name: str, port: int = 8080) -> None:
        """Start a Trame control panel app."""
        control_panel_dir = self.project_root / "control_panel"
        if not control_panel_dir.exists():
            print(f"Control panel directory not found: {control_panel_dir}")
            return

        # Start in new process group so we can kill all child processes
        process = subprocess.Popen(
            [sys.executable, "-m", f"apps.{app_name}", "--port", str(port)],
            cwd=control_panel_dir,
            start_new_session=True,
        )
        self.processes.append(ServerProcess(name=f"trame-{app_name}", process=process))
        print(f"Control panel {app_name} started on http://localhost:{port}")

    def wait(self) -> None:
        """Wait for all processes to complete."""
        for server in self.processes:
            server.process.wait()

    def shutdown(self) -> None:
        """Terminate all running processes."""
        # Prevent recursive calls from multiple Ctrl+C
        if self._shutting_down:
            return
        self._shutting_down = True

        print("\nShutting down servers...")
        import os as _os

        for server in self.processes:
            if server.process.poll() is None:
                try:
                    # Kill entire process group (includes child processes)
                    pgid = _os.getpgid(server.process.pid)
                    _os.killpg(pgid, signal.SIGTERM)
                except (ProcessLookupError, OSError):
                    # Process already gone or no process group
                    server.process.terminate()

        # Wait with timeout, then force kill if necessary
        for server in self.processes:
            if server.process.poll() is not None:
                print(f"Stopped {server.name}")
                continue
            try:
                server.process.wait(timeout=3)
                print(f"Stopped {server.name}")
            except subprocess.TimeoutExpired:
                print(f"Force killing {server.name}...")
                try:
                    pgid = _os.getpgid(server.process.pid)
                    _os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    server.process.kill()
                try:
                    server.process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass  # Give up, exit anyway
                print(f"Killed {server.name}")


def main() -> None:
    """Entry point for the server manager."""
    parser = argparse.ArgumentParser(description="Start SBI development servers")
    parser.add_argument(
        "--django-port", type=int, default=8000, help="Django server port"
    )
    parser.add_argument(
        "--frontend-port", type=int, default=5173, help="Frontend dev server port"
    )
    parser.add_argument(
        "--no-frontend", action="store_true", help="Skip starting frontend server"
    )
    parser.add_argument(
        "--no-django", action="store_true", help="Skip starting Django server"
    )
    parser.add_argument(
        "--trame", type=str, help="Control panel app to start (e.g., result_viewer)"
    )
    parser.add_argument(
        "--trame-port", type=int, default=8080, help="Trame server port"
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.resolve()
    manager = ServerManager(project_root)

    def signal_handler(_sig: int, _frame: object) -> None:
        if manager._shutting_down:
            # Second Ctrl+C - force exit immediately
            print("\nForce exit...")
            sys.exit(1)
        manager.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if not args.no_django:
        manager.start_django(port=args.django_port)

    if not args.no_frontend:
        manager.start_frontend(port=args.frontend_port)

    if args.trame:
        manager.start_trame(app_name=args.trame, port=args.trame_port)

    if not manager.processes:
        print("No servers started. Use --help for options.")
        return

    print("\nPress Ctrl+C to stop all servers")
    manager.wait()


if __name__ == "__main__":
    main()
