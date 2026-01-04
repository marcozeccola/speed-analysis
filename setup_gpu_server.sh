#!/bin/bash
# GPU Server Setup Script for Speed Analysis Project
# Run this on your GPU server after cloning the repository

set -e  # Exit on error

echo "🚀 Speed Analysis - GPU Server Setup"
echo "===================================="

# Check if CUDA is available
echo ""
echo "📊 Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo "✅ NVIDIA GPU detected"
else
    echo "⚠️  WARNING: nvidia-smi not found. Make sure NVIDIA drivers are installed!"
fi

# Python version check
echo ""
echo "🐍 Checking Python version..."
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install GPU requirements
echo ""
echo "📥 Installing GPU-optimized dependencies..."
cd server
pip install -r requirements-gpu.txt

# Verify PyTorch GPU support
echo ""
echo "🔍 Verifying PyTorch GPU support..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}' if torch.cuda.is_available() else 'No CUDA')"

if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "✅ PyTorch GPU support confirmed!"
else
    echo "⚠️  WARNING: PyTorch installed but CUDA not available!"
    echo "   Check CUDA drivers and compatibility."
fi

# Install Redis if not present
echo ""
echo "🔴 Checking Redis..."
if command -v redis-server &> /dev/null; then
    echo "✅ Redis is installed"
else
    echo "📥 Installing Redis..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y redis-server
    elif command -v yum &> /dev/null; then
        sudo yum install -y redis
    else
        echo "⚠️  Please install Redis manually"
    fi
fi

# Start Redis
echo ""
echo "🔴 Starting Redis..."
sudo systemctl start redis-server 2>/dev/null || redis-server --daemonize yes

# Create startup scripts
echo ""
echo "📝 Creating startup scripts..."
cd ..

# Create start script
cat > start_gpu.sh << 'EOF'
#!/bin/bash
# Start all services for GPU deployment

echo "🚀 Starting Speed Analysis Server (GPU Mode)"

# Activate virtual environment
source venv/bin/activate

# Start Redis
echo "🔴 Starting Redis..."
redis-server --daemonize yes

# Start Celery worker
echo "⚙️  Starting Celery worker..."
cd server
celery -A tasks worker --loglevel=info --concurrency=1 &
CELERY_PID=$!

# Start FastAPI server
echo "🌐 Starting FastAPI server..."
uvicorn main:app --host 0.0.0.0 --port 8000 &
UVICORN_PID=$!

echo ""
echo "✅ All services started!"
echo "📊 API: http://localhost:8000"
echo "📈 Viewer: http://localhost:8000/viewer"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for Ctrl+C
trap "echo 'Stopping services...'; kill $CELERY_PID $UVICORN_PID; exit" INT
wait
EOF

chmod +x start_gpu.sh

# Create stop script
cat > stop_gpu.sh << 'EOF'
#!/bin/bash
# Stop all services

echo "🛑 Stopping Speed Analysis Server..."

# Kill Celery
pkill -f "celery -A tasks worker"

# Kill uvicorn
pkill -f "uvicorn main:app"

# Stop Redis (optional)
# redis-cli shutdown

echo "✅ All services stopped"
EOF

chmod +x stop_gpu.sh

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Copy your 'best.pt' model file to server/ directory"
echo "   2. Run: ./start_gpu.sh"
echo "   3. Open: http://YOUR_SERVER_IP:8000/viewer"
echo ""
echo "🔧 Useful commands:"
echo "   Start services:  ./start_gpu.sh"
echo "   Stop services:   ./stop_gpu.sh"
echo "   Check GPU:       nvidia-smi"
echo "   View logs:       tail -f server/celery.log"
