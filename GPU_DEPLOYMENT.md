# Speed Analysis - GPU Server Deployment Guide

## 🎯 Recommended: Vast.ai Setup (Most Cost-Effective)

### 1. Create Vast.ai Account
- Go to https://vast.ai
- Sign up and add $10-20 credit
- Browse available GPUs (RTX 3090, RTX 4090, or A6000 recommended)

### 2. Rent a GPU Instance
- **Search**: Look for "pytorch" template or Ubuntu 22.04 with CUDA
- **Filter**: 
  - GPU: RTX 3090 or better
  - CUDA: 11.8 or 12.x
  - Disk: 50GB minimum
  - Price: $0.20-0.50/hour
- **Click**: "Rent" on your chosen instance

### 3. Connect via SSH
```bash
ssh -p [PORT] root@[IP_ADDRESS] -L 8000:localhost:8000
```
(Copy exact command from Vast.ai dashboard)

### 4. Deploy the Project
```bash
# Clone your repository (or upload via SCP)
git clone https://github.com/YOUR_USERNAME/speed-analysis.git
cd speed-analysis

# Run setup script
chmod +x setup_gpu_server.sh
./setup_gpu_server.sh

# Copy your model file
scp -P [PORT] server/best.pt root@[IP_ADDRESS]:~/speed-analysis/server/

# Start services
./start_gpu.sh
```

### 5. Access the Application
- Open in browser: http://localhost:8000/viewer
- The SSH tunnel (-L 8000:localhost:8000) forwards the remote port to your local machine

---

## 🔥 Expected Performance

| Environment | Time per Frame | Total (1000 frames) | Speedup |
|-------------|----------------|---------------------|---------|
| **CPU (Intel i7)** | ~380ms | ~6 minutes | 1x |
| **GPU (RTX 3090)** | ~20-30ms | ~30 seconds | **12-15x** |
| **GPU (RTX 4090)** | ~15-20ms | ~20 seconds | **20x** |

---

## 🛠️ Alternative: RunPod Setup

1. Go to https://www.runpod.io
2. Create account and add credits
3. Deploy → GPU Instance → PyTorch template
4. SSH connect and follow same deployment steps

---

## ☁️ Alternative: Google Colab (Free but Limited)

### Upload as Colab Notebook:

```python
# Install dependencies
!pip install ultralytics opencv-python mediapipe fastapi uvicorn celery redis pykalman

# Upload your files
from google.colab import files
uploaded = files.upload()  # Upload best.pt

# Start Redis in background
!apt-get install -y redis-server
!redis-server --daemonize yes

# Run your analysis
# (Interactive mode - not for production)
```

---

## 🔧 Troubleshooting

### GPU Not Detected
```bash
# Check NVIDIA drivers
nvidia-smi

# Verify PyTorch GPU
python3 -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with correct CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Port Already in Use
```bash
# Find and kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use different port
uvicorn main:app --host 0.0.0.0 --port 8080
```

### Redis Connection Error
```bash
# Start Redis
sudo systemctl start redis-server
# Or
redis-server --daemonize yes

# Test connection
redis-cli ping
```

---

## 💰 Cost Estimates

| Provider | GPU | Cost/Hour | 1 Hour Analysis (~120 videos) |
|----------|-----|-----------|-------------------------------|
| **Vast.ai** | RTX 3090 | $0.20-0.30 | **$0.20-0.30** |
| **RunPod** | RTX 3090 | $0.34 | $0.34 |
| **AWS** | g4dn.xlarge | $0.526 | $0.53 |
| **GCP** | T4 | $0.35 | $0.35 |
| **Azure** | NC6 | $0.90 | $0.90 |

**Recommendation**: Start with Vast.ai for best price/performance ratio.

---

## 📊 Monitoring

### Check GPU Usage
```bash
watch -n 1 nvidia-smi
```

### View Logs
```bash
# Celery worker logs
tail -f celery.log

# API logs
tail -f uvicorn.log
```

### Performance Test
```bash
# Single video analysis should take ~20-30 seconds
curl -X POST http://localhost:8000/api/analyze-videos/ \
  -F "video_a=@test_video.mp4"
```

---

## 🔒 Security Notes

- **Never expose Redis to public internet** (use firewall)
- **Use SSH tunneling** for accessing the web interface remotely
- **Set strong passwords** if deploying in production
- **Use HTTPS** in production (add nginx reverse proxy)

---

## 📝 Quick Commands

```bash
# Start all services
./start_gpu.sh

# Stop all services
./stop_gpu.sh

# Restart Celery only
pkill -f celery
cd server && celery -A tasks worker --loglevel=info &

# Check service status
ps aux | grep -E "celery|uvicorn|redis"

# Monitor GPU in real-time
nvidia-smi dmon
```

---

## 🎓 Next Steps

1. **Test locally first** to ensure everything works
2. **Backup your model file** (best.pt)
3. **Choose a GPU provider** (Vast.ai recommended)
4. **Deploy and test** with a small video
5. **Monitor costs** through provider dashboard
6. **Scale up** as needed

**Ready?** Run `./setup_gpu_server.sh` on your GPU server! 🚀
