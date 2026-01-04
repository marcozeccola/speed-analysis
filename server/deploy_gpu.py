"""
GPU Deployment Configuration
Add this to the top of tasks.py or use as reference
"""

import torch

# Check GPU availability
def check_gpu():
    if torch.cuda.is_available():
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return True
    else:
        print("❌ No GPU available, using CPU")
        return False

# Device configuration
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# GPU-optimized settings for YOLO
GPU_YOLO_SETTINGS = {
    'device': DEVICE,
    'half': True if DEVICE.startswith('cuda') else False,  # FP16 for faster inference on GPU
    'verbose': False,
    'conf': 0.25,
    'iou': 0.40,
}

# Instructions for deployment
DEPLOYMENT_CHECKLIST = """
📋 GPU DEPLOYMENT CHECKLIST:

1. ✅ Install CUDA-enabled PyTorch:
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

2. ✅ Verify GPU in Python:
   python -c "import torch; print(torch.cuda.is_available())"

3. ✅ Update tasks.py model loading:
   model = YOLO(model_path)
   if torch.cuda.is_available():
       model.to('cuda:0')

4. ✅ Update model.track() call:
   res = model.track(frame, device='cuda:0', half=True, ...)

5. ✅ Expected Speedup:
   CPU: ~380ms/frame → GPU: ~20-30ms/frame (10-15x faster!)

6. 🌐 Deployment Options:
   
   A. VAST.AI (Recommended):
      - Go to: https://vast.ai
      - Search: "RTX 3090" or "A6000"
      - Rent: ~$0.20-0.50/hour
      - Install: Docker template or manual setup
   
   B. RUNPOD:
      - Go to: https://www.runpod.io
      - Choose: GPU pod
      - Deploy: Template with CUDA + Python
   
   C. GOOGLE COLAB:
      - Upload code to Colab
      - Runtime → Change runtime type → GPU
      - Install dependencies in notebook
   
   D. PAPERSPACE:
      - Sign up at: https://www.paperspace.com
      - Create: GPU instance (P5000 or better)
      - SSH and deploy

7. 📦 requirements.txt for GPU:
   torch>=2.0.0
   torchvision>=0.15.0
   ultralytics>=8.0.0
   opencv-python>=4.8.0
   mediapipe>=0.10.0
   celery>=5.3.0
   redis>=4.5.0
   fastapi>=0.100.0
   uvicorn>=0.23.0
   pykalman
   scipy
   matplotlib
"""

if __name__ == "__main__":
    check_gpu()
    print(DEPLOYMENT_CHECKLIST)
