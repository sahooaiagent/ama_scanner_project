import os
import subprocess
import datetime
import json
import asyncio
import signal
from typing import List, Optional
from pathlib import Path
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.memory import MemoryJobStore

# Resolve scanner path relative to this file (project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCANNER_PATH = str(PROJECT_ROOT / "New_AMA_Pro_Scanner.py")

app = FastAPI(title="AMA Pro Scanner Dashboard")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Scheduler
jobstores = {
    'default': MemoryJobStore()
}
scheduler = AsyncIOScheduler(jobstores=jobstores)

LOG_FILE = str(PROJECT_ROOT / "scanner_run.log")
# Track running processes
running_process = None

class ScanRequest(BaseModel):
    exchange: str = "binance"
    timeframes: List[str]
    symbol_limit: int
    schedule_time: Optional[str] = None # ISO format string

async def run_scanner(exchange: str, timeframes: List[str], symbol_limit: int):
    """Executes the scanner script using asyncio."""
    global running_process

    tf_str = ",".join(timeframes)
    cmd = [
        "python3", SCANNER_PATH,
        "--exchange", exchange,
        "--symbols", str(symbol_limit),
        "--timeframes", tf_str,
        "--no-websocket", "--no-ml", "--no-redis"
    ]

    with open(LOG_FILE, "a") as log:
        log.write(f"\n🔄 SCAN IN PROGRESS - Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"Parameters: Symbols={symbol_limit}, Timeframes={tf_str}\n")
        log.flush()

        try:
            # Use asyncio.create_subprocess_exec for non-blocking execution
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=log,
                stderr=log
            )
            running_process = process
            await process.wait()
            log.write(f"\n✅ SCAN COMPLETED - Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        except Exception as e:
            log.write(f"\n❌ ERROR: Scan failed with exception: {str(e)}\n")
        finally:
            running_process = None
            log.flush()

@app.on_event("startup")
async def startup_event():
    scheduler.start()

@app.post("/scan")
async def trigger_scan(request: ScanRequest, background_tasks: BackgroundTasks):
    if request.schedule_time:
        try:
            run_at = datetime.datetime.fromisoformat(request.schedule_time)
            if run_at < datetime.datetime.now():
                raise HTTPException(status_code=400, detail="Schedule time must be in the future")

            job_id = f"scan_{int(run_at.timestamp())}"
            scheduler.add_job(
                run_scanner,
                'date',
                run_date=run_at,
                args=[request.exchange, request.timeframes, request.symbol_limit],
                id=job_id
            )
            return {"status": "scheduled", "job_id": job_id, "run_at": str(run_at)}
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format")
    else:
        background_tasks.add_task(run_scanner, request.exchange, request.timeframes, request.symbol_limit)
        return {"status": "started"}

@app.get("/results")
async def get_results():
    """Fetches the latest scan results from scan_results.json."""
    results_file = PROJECT_ROOT / "scan_results.json"
    if not results_file.exists():
        return {"results": [], "scan_time": None}

    try:
        with open(results_file, "r") as f:
            data = json.load(f)

        # Map signal fields to frontend expected names
        results = []
        for s in data.get("signals", []):
            results.append({
                "Crypto Name": s.get("symbol", ""),
                "Timeperiod": s.get("timeframe", ""),
                "Signal": s.get("signal_type", ""),
                "Angle": s.get("angle", 0),
                "Daily Change": s.get("daily_change", 0),
                "Timestamp": s.get("timestamp", ""),
                "Price": s.get("price", 0),
                "Confidence": s.get("confidence", 0),
                "Regime": s.get("regime", ""),
                "Exchange": s.get("exchange", ""),
                "Volume": s.get("volume", 0),
            })

        return {"results": results, "scan_time": data.get("scan_time")}
    except Exception as e:
        print(f"Error reading scan_results.json: {e}")
        return {"results": [], "scan_time": None}

@app.post("/clear-logs")
async def clear_logs():
    """Clears the scanner log file."""
    try:
        with open(LOG_FILE, "w") as f:
            f.write(f"--- Logs cleared at {datetime.datetime.now()} ---\n")
        return {"status": "success", "message": "Logs cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/logs")
async def get_logs():
    if not os.path.exists(LOG_FILE):
        return {"logs": "No logs available."}
    with open(LOG_FILE, "r") as f:
        return {"logs": f.read()[-5000:]} # Return last 5000 chars

@app.get("/config")
async def get_config():
    return {
        "available_timeframes": ["3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"],
        "available_symbol_limits": [20, 50, 100, 200, 300, 400, 500]
    }

@app.get("/status")
async def get_status():
    jobs = []
    for job in scheduler.get_jobs():
        # Extract job details from args
        exchange = job.args[0] if len(job.args) > 0 else "unknown"
        timeframes = job.args[1] if len(job.args) > 1 else []
        symbol_limit = job.args[2] if len(job.args) > 2 else 0

        jobs.append({
            "id": job.id,
            "next_run_time": str(job.next_run_time),
            "exchange": exchange,
            "timeframes": timeframes,
            "symbol_limit": symbol_limit
        })
    return {
        "jobs": jobs,
        "scan_running": running_process is not None
    }

@app.post("/cancel-job/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a scheduled job."""
    try:
        scheduler.remove_job(job_id)
        return {"status": "success", "message": f"Job {job_id} cancelled"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Job not found: {str(e)}")

@app.post("/stop-scan")
async def stop_scan():
    """Stops the currently running scan."""
    global running_process
    if running_process is None:
        raise HTTPException(status_code=404, detail="No scan is currently running")

    try:
        running_process.terminate()
        await asyncio.sleep(1)
        if running_process.returncode is None:
            running_process.kill()

        with open(LOG_FILE, "a") as log:
            log.write(f"\n⚠️ SCAN STOPPED - User requested stop at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        running_process = None
        return {"status": "stopped", "message": "Scan stopped successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop scan: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
