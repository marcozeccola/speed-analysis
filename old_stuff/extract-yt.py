# ...existing code...
"""
Estrai N frame (immagini) da un video YouTube dato il link.

Modifica le variabili YT_URL e OUT_DIR qui sotto, poi esegui:
    python extract-yt.py

Dipendenze:
    pip install pytube opencv-python

Nota: il video viene scaricato temporaneamente nella cartella temporanea del sistema
e rimosso al termine (se possibile).
"""
import os
import cv2
import tempfile
import shutil
from pytube import YouTube

# ------------------ Variabili modificabili ------------------
# Link al video YouTube:
YT_URL = "https://www.youtube.com/watch?v=gjB-hV3at0s"

# Cartella dove salvare i frame estratti:
OUT_DIR = r"C:\Users\marco\OneDrive\Documenti\speed-analysis\yt_frames"

# Prefisso dei file immagine:
NAME = "yt_"

# Numero di frame da estrarre (spaziati uniformemente):
NUM_FRAMES = 50

# Estensione immagine ("jpg" o "png"):
EXT = "jpg"

# Ridimensionamento opzionale: (width, height) o None per mantenere la risoluzione originale
RESIZE = None  # es.: (640, 360)
# --------------------------------------------------------------

def download_youtube_video(url: str, tmp_dir: str) -> str:
    yt = YouTube(url)
    # scegli lo stream progressivo mp4 con risoluzione maggiore disponibile
    stream = yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first()
    if stream is None:
        raise RuntimeError("Nessuno stream mp4 progressivo trovato per questo video.")
    out_path = stream.download(output_path=tmp_dir)
    return out_path

def extract_frames_from_file(video_path: str, out_dir: str, count: int, ext: str = "jpg", resize: tuple = None, name_prefix: str = "frame_"):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total == 0:
        cap.release()
        raise RuntimeError("Il video non contiene frame o non è possibile leggere il conteggio frame.")

    count = max(1, int(count))
    if count >= total:
        indices = list(range(total))
    else:
        if count == 1:
            indices = [total // 2]
        else:
            indices = [int(round(i * (total - 1) / (count - 1))) for i in range(count)]

    pad = len(str(len(indices)))
    saved = 0
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        if resize:
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
        out_name = f"{name_prefix}{i+1:0{pad}d}.{ext}"
        out_path = os.path.join(out_dir, out_name)
        cv2.imwrite(out_path, frame)
        saved += 1

    cap.release()
    return saved

if __name__ == "__main__":
    tmp_dir = tempfile.mkdtemp(prefix="yt_download_")
    video_file = None
    try:
        print("Scaricamento video...")
        video_file = download_youtube_video(YT_URL, tmp_dir)
        print("Scaricato in:", video_file)
        saved = extract_frames_from_file(video_file, OUT_DIR, NUM_FRAMES, ext=EXT, resize=RESIZE, name_prefix=NAME)
        print(f"Salvati {saved} frame in {os.path.abspath(OUT_DIR)}")
    except Exception as e:
        print("Errore:", e)
        raise
    finally:
        # prova a rimuovere i file temporanei
        try:
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
        except Exception:
            pass
# ...existing code...