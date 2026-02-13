#!/bin/bash

echo "Starting AMA Pro Scanner Modern Dashboard..."

# Install dependencies if needed (optional, up to user)
# pip install -r web_dashboard/requirements_web.txt

# Start Backend in background
echo "Launching Backend (FastAPI) on port 8000..."
python3 web_dashboard/backend/main.py > web_dashboard/backend_status.log 2>&1 &
BACKEND_PID=$!

# Start Frontend (simple python server) on port 8080
echo "Launching Frontend on http://localhost:8080 ..."
cd web_dashboard/frontend && python3 -m http.server 8080 > ../frontend_status.log 2>&1 &
FRONTEND_PID=$!

echo "Dashboard is running!"
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo "Press Ctrl+C to stop both (or kill them manually)."

# Wait for Ctrl+C
trap "kill $BACKEND_PID $FRONTEND_PID; echo 'Stopped.'; exit" INT
wait
