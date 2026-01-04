from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from celery.result import AsyncResult
from typing import Dict, Any
import os
import tempfile

from .tasks import analyze_climbing_videos_task, app as celery_app
from ultralytics import YOLO

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],             # Allows all origins (including 'null' from local files)
    allow_credentials=True,
    allow_methods=["*"],             # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],             # Allows all headers
)
# Crea cartella temporanea per i video
UPLOAD_DIR = tempfile.mkdtemp(prefix="video_uploads_")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Path al viewer HTML
VIEWER_PATH = os.path.join(os.path.dirname(__file__), "..", "client", "analysis_viewer.html")

PATH_TO_MODEL = r"best.pt"
try:
    main_model = YOLO(PATH_TO_MODEL)
    from .tasks import set_model
    set_model(main_model)
except Exception as e:
    print(f"Warning: Could not load model: {e}")

@app.get("/api/health")
def health_check():
    """ Verifica che l'API sia attiva. """
    return {"status": "ok", "service": "FastAPI"}

@app.get("/viewer")
def viewer(job_id: str = ""):
    """ Serve the analysis viewer page (optionally with job_id in URL). """
    return FileResponse(VIEWER_PATH, media_type="text/html")

@app.post("/api/analyze-videos/")
async def create_analysis(video_a: UploadFile = File(...), video_b: UploadFile = File(None) ):
    """ Endpoint per avviare l'analisi dei video. """
    
    try:
        # Salva i video temporaneamente
        video_a_path = os.path.join(UPLOAD_DIR, f"video_a_{video_a.filename}")
        
        with open(video_a_path, "wb") as f:
            contents = await video_a.read()
            f.write(contents)
        
        video_b_path = None
        if video_b:
            video_b_path = os.path.join(UPLOAD_DIR, f"video_b_{video_b.filename}")
            with open(video_b_path, "wb") as f:
                contents = await video_b.read()
                f.write(contents)
        
        # Invia la task a Celery con i percorsi dei video
        task = analyze_climbing_videos_task.delay(video_a_path, video_b_path)
        
        # Risposta immediata
        return {
            "job_id": task.id, 
            "status": "processing", 
            "message": "Analisi video avviata."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error uploading videos: {str(e)}")


@app.get("/api/analysis-status/{job_id}")
async def get_analysis_status(job_id: str) -> Dict[str, Any]:
    """ Controlla lo stato del job e restituisce i risultati se pronto. """
    
    # Usa l'app Celery importata per controllare lo stato
    task = AsyncResult(job_id, app=celery_app)
    
    if task.ready():
        # If task succeeded, try to return the result; otherwise stringify it to avoid
        # Pydantic serialization errors for exceptions or non-serializable objects.
        if task.successful():
            result = task.result
            try:
                return {"status": task.state, "data": result}
            except Exception:
                return {"status": task.state, "data": str(result)}
        else:
            # Task finished but failed or was revoked. Return a stringified error.
            err = task.result if task.result is not None else getattr(task, 'traceback', None)
            return {"status": task.state, "error": str(err)}
    elif task.state in ['PENDING', 'STARTED', 'RETRY']:
        return {"status": task.state}
    else:
        raise HTTPException(status_code=400, detail=f"Job {job_id} failed with state {task.state}")