from celery import Celery
import time
import random
 
REDIS_URL = "redis://localhost:6379/0" 
REDIS_BACKEND = "redis://localhost:6379/1" 

app = Celery(
    'video_tasks', 
    broker=REDIS_URL,
    backend=REDIS_BACKEND
)

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
)

@app.task(name='analyze_climbing_videos')
def analyze_climbing_videos_task(filename_a: str, filename_b: str) -> dict:
    """ 
    Simula l'analisi video: attende e restituisce dati fittizi.
    I nomi dei file sono solo per debug.
    """
    print(f"--- INIZIO ELABORAZIONE MOCK: {filename_a} vs {filename_b} ---")
    
    # Simula l'elaborazione time consuming
    delay = random.randint(1, 4)
    time.sleep(delay)

    # Genera dati mock
    time_points = list(range(10))  
     
    mock_data_A = {
        "pos_Y": [random.uniform(0.5, 3.0) for _ in time_points],
        "vel_Y": [random.uniform(-1.0, 1.0) for _ in time_points],
        "acc_Y": [random.uniform(-5.0, 5.0) for _ in time_points],
        "time": time_points,
    }
    
    mock_data_B = {
        "pos_Y": [random.uniform(0.5, 3.0) for _ in time_points],
        "vel_Y": [random.uniform(-1.0, 1.0) for _ in time_points],
        "acc_Y": [random.uniform(-5.0, 5.0) for _ in time_points],
        "time": time_points,
    }
    
    print(f"--- ELABORAZIONE MOCK COMPLETATA in {delay} secondi. ---")
 
    return {
        "climber_A": mock_data_A,
        "climber_B": mock_data_B,
        "processing_time": delay
    }