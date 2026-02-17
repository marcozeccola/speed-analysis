from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from celery.result import AsyncResult
from typing import Dict, Any
import os
import re
import tempfile

from server.tasks import analyze_climbing_videos_task, app as celery_app

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Usa la directory temporanea del sistema (compatibile Windows/Linux)
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "climbing_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

VIEWER_PATH = os.path.join(os.path.dirname(__file__), "..", "client", "analysis_viewer.html")

@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "FastAPI"}

@app.get("/viewer")
def viewer(job_id: str = ""):
    return FileResponse(VIEWER_PATH, media_type="text/html")

@app.post("/api/analyze-videos/")
async def create_analysis(video_a: UploadFile = File(...), video_b: UploadFile = File(None)):
    try:
        # Crea una directory temporanea univoca per questa richiesta
        request_dir = tempfile.mkdtemp(prefix="video_uploads_", dir=UPLOAD_DIR)
        
        # Sanitizza filename (rimuovi spazi e caratteri speciali)
        safe_name_a = re.sub(r'[^a-zA-Z0-9._-]', '_', video_a.filename)
        video_a_path = os.path.join(request_dir, f"video_a_{safe_name_a}")
        
        with open(video_a_path, "wb") as f:
            contents = await video_a.read()
            f.write(contents)
        
        video_b_path = None
        if video_b:
            safe_name_b = re.sub(r'[^a-zA-Z0-9._-]', '_', video_b.filename)
            video_b_path = os.path.join(request_dir, f"video_b_{safe_name_b}")
            with open(video_b_path, "wb") as f:
                contents = await video_b.read()
                f.write(contents)
        
        # Verifica che i file esistano
        if not os.path.exists(video_a_path):
            raise HTTPException(status_code=500, detail=f"Failed to save video A")
        if video_b_path and not os.path.exists(video_b_path):
            raise HTTPException(status_code=500, detail=f"Failed to save video B")
        
        task = analyze_climbing_videos_task.delay(video_a_path, video_b_path)
        return {"job_id": task.id, "status": "processing", "message": "Analisi video avviata."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error uploading videos: {str(e)}")

@app.get("/api/analysis-status/{job_id}")
async def get_analysis_status(job_id: str):
    task = AsyncResult(job_id, app=celery_app)
    
    if task.ready():
        if task.successful():
            return {"status": "SUCCESS", "data": task.result}
        else:
            return {"status": task.state, "error": str(task.result)}
    elif task.state == 'PENDING':
        return {"status": "PENDING", "message": "Your request is in queue.", "progress": 0}
    elif task.state == 'STARTED':
        return {"status": "STARTED", "message": "Analysis started...", "progress": 0}
    elif task.state == 'PROGRESS':
        info = task.info or {}
        return {"status": "PROGRESS", "current": info.get('current', 0), "total": info.get('total', 100), "progress": info.get('progress', 0), "message": info.get('message', 'Processing...')}
    elif task.state == 'RETRY':
        return {"status": "RETRY", "message": "Retrying..."}
    else:
        raise HTTPException(status_code=400, detail=f"Job failed: {task.state}")
