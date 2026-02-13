import os
import subprocess
import datetime
import json
import asyncio
from typing import List, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.memory import MemoryJobStore
import pandas as pd
import glob

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

LOG_FILE = "../scanner_run.log"
RESULTS_DIR = ".." # Results are saved as CSVs in the root by the scanner

class ScanRequest(BaseModel):
    timeframes: List[str]
    symbol_limit: int
    schedule_time: Optional[str] = None # ISO format string

async def run_scanner(timeframes: List[str], symbol_limit: int):
    """Executes the scanner script using asyncio."""
    tf_str = ",".join(timeframes)
    cmd = [
        "python3", "ama_pro_scanner.py", 
        str(symbol_limit), 
        "--timeframes", tf_str
    ]
    
    with open(LOG_FILE, "a") as log:
        log.write(f"\n--- Starting Scan: {datetime.datetime.now()} ---\n")
        log.write(f"Parameters: Symbols={symbol_limit}, Timeframes={tf_str}\n")
        log.flush()
        
        try:
            # Use asyncio.create_subprocess_exec for non-blocking execution
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=log,
                stderr=log
            )
            await process.wait()
            log.write(f"--- Scan Completed: {datetime.datetime.now()} ---\n")
        except Exception as e:
            log.write(f"--- ERROR: Scan failed with exception: {str(e)} ---\n")
        finally:
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
                args=[request.timeframes, request.symbol_limit],
                id=job_id
            )
            return {"status": "scheduled", "job_id": job_id, "run_at": str(run_at)}
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format")
    else:
        background_tasks.add_task(run_scanner, request.timeframes, request.symbol_limit)
        return {"status": "started"}

@app.get("/results")
async def get_results():
    """Fetches the latest scan results from CSV files."""
    csv_files = glob.glob("../ama_pro_scan_results_*.csv")
    if not csv_files:
        return {"results": []}
    
    # Sort files by modification time, latest first
    csv_files.sort(key=os.path.getmtime, reverse=True)
    
    all_results = []
    # Optionally limit to the last few scan files to keep payload reasonable
    for latest_file in csv_files[:5]: 
        try:
            df = pd.read_csv(latest_file)
            all_results.extend(df.to_dict(orient="records"))
        except Exception as e:
            print(f"Error reading {latest_file}: {e}")
            
    return {"results": all_results}

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
        "available_timeframes": ["15m", "30m", "45m", "1h", "2h", "4h", "12h", "1d", "2d", "1w", "1M"],
        "available_symbol_limits": [20, 50, 100, 200, 300, 400, 500]
    }

@app.get("/status")
async def get_status():
    jobs = []
    for job in scheduler.get_jobs():
        jobs.append({
            "id": job.id,
            "next_run_time": str(job.next_run_time)
        })
    return {"jobs": jobs}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
