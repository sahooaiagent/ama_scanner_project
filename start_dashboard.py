#!/usr/bin/env python3
"""
AMA Pro Scanner Dashboard Launcher
Starts the backend server and opens the dashboard in your browser with a single command.
"""

import subprocess
import time
import webbrowser
import os
import sys
import signal
from pathlib import Path

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header():
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}")
    print("  🚀 AMA Pro Scanner Dashboard Launcher")
    print(f"{'='*60}{Colors.ENDC}\n")

def check_port(port=8000):
    """Check if the port is already in use."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def main():
    print_header()

    # Get the script directory
    script_dir = Path(__file__).parent.absolute()
    backend_dir = script_dir / "web_dashboard" / "backend"
    frontend_file = script_dir / "web_dashboard" / "frontend" / "index.html"

    # Check if directories exist
    if not backend_dir.exists():
        print(f"{Colors.FAIL}❌ Error: Backend directory not found at {backend_dir}{Colors.ENDC}")
        sys.exit(1)

    if not frontend_file.exists():
        print(f"{Colors.FAIL}❌ Error: Frontend file not found at {frontend_file}{Colors.ENDC}")
        sys.exit(1)

    # Check if port 8000 is already in use
    if check_port(8000):
        print(f"{Colors.WARNING}⚠️  Port 8000 is already in use.{Colors.ENDC}")
        print(f"{Colors.OKCYAN}   The backend might already be running.{Colors.ENDC}")
        print(f"{Colors.OKCYAN}   Opening dashboard in browser...{Colors.ENDC}\n")
        webbrowser.open(f'file://{frontend_file}')
        return

    # Start the backend server
    print(f"{Colors.OKBLUE}🔧 Starting FastAPI backend server...{Colors.ENDC}")
    backend_process = subprocess.Popen(
        [sys.executable, "main.py"],
        cwd=str(backend_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Wait for the server to start
    print(f"{Colors.OKCYAN}⏳ Waiting for server to be ready...{Colors.ENDC}")
    max_retries = 30
    for i in range(max_retries):
        if check_port(8000):
            print(f"{Colors.OKGREEN}✅ Backend server started successfully!{Colors.ENDC}\n")
            break
        time.sleep(0.5)
    else:
        print(f"{Colors.FAIL}❌ Error: Server failed to start within 15 seconds{Colors.ENDC}")
        backend_process.terminate()
        sys.exit(1)

    # Open the dashboard in the default browser
    print(f"{Colors.OKBLUE}🌐 Opening dashboard in your browser...{Colors.ENDC}\n")
    webbrowser.open(f'file://{frontend_file}')

    print(f"{Colors.OKGREEN}{Colors.BOLD}{'='*60}")
    print("  ✅ Dashboard is now running!")
    print(f"{'='*60}{Colors.ENDC}\n")
    print(f"{Colors.OKCYAN}📊 Dashboard URL: file://{frontend_file}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}🔌 Backend API:   http://localhost:8000{Colors.ENDC}")
    print(f"\n{Colors.WARNING}Press Ctrl+C to stop the server{Colors.ENDC}\n")

    # Keep the script running and handle Ctrl+C
    def signal_handler(sig, frame):
        print(f"\n\n{Colors.WARNING}🛑 Shutting down...{Colors.ENDC}")
        backend_process.terminate()
        try:
            backend_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            backend_process.kill()
        print(f"{Colors.OKGREEN}✅ Server stopped successfully{Colors.ENDC}\n")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Wait for the backend process
    try:
        backend_process.wait()
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == "__main__":
    main()
