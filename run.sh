#!/bin/bash
# run.sh — Start the Microplastics Detector web app

set -e

if [ -z "$VIRTUAL_ENV" ]; then
  echo "⚠️  Virtual environment not active. Run: source venv/bin/activate"
  exit 1
fi

pip install fastapi uvicorn python-multipart --quiet

echo ""
echo "🔬  Microplastics Detector"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🌐  Open in browser → http://localhost:8000"
echo "🛑  Stop with Ctrl+C"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# cd into the script's own directory so Python can find all modules
cd "$(dirname "$0")"

python server/app.py