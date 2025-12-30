from fastapi import FastAPI, UploadFile, File, HTTPException
from celery.result import AsyncResult
from typing import Dict, Any

from .tasks import analyze_climbing_videos_task, app as celery_app

app = FastAPI()


PATH_TO_MODEL = r"path/to/model"
main_model = YOLO(PATH_TO_MODEL)
set_model( main_model )

@app.get("/api/health")
def health_check():
    """ Verifica che l'API sia attiva. """
    return {"status": "ok", "service": "FastAPI"}

@app.post("/api/analyze-videos/")
async def create_analysis(video_a: UploadFile = File(...), video_b: UploadFile = File(...)):
    """ Endpoint per avviare l'analisi (simulata). """
    
    #Simula la gestione dei file: usiamo solo i nomi per la task
    filename_a = video_a.filename
    filename_b = video_b.filename
    
    #Invia la task a Celery
    task = analyze_climbing_videos_task.delay(filename_a, filename_b)
    
    #Risposta immediata
    return {"job_id": task.id, "status": "processing", "message": "Analisi simulata avviata."}


@app.get("/api/analysis-status/{job_id}")
async def get_analysis_status(job_id: str) -> Dict[str, Any]:
    """ Controlla lo stato del job e restituisce i risultati se pronto. """
    
    # Usa l'app Celery importata per controllare lo stato
    task = AsyncResult(job_id, app=celery_app)
    
    if task.ready():
        return {"status": task.state, "data": task.result}
    elif task.state in ['PENDING', 'STARTED', 'RETRY']:
        return {"status": task.state}
    else:
        raise HTTPException(status_code=400, detail=f"Job {job_id} failed with state {task.state}")