#!/usr/bin/env bash
set -e

# ------- 1️⃣ Launch Flask (backend) -------
echo "▶️ Starting Flask backend on :5000"
python /app/backend/clip_api.py &
FLASK_PID=$!

# ------- 2️⃣ Launch Next.js (frontend) -------
echo "▶️ Starting Next.js frontend on :3000"
cd /app/frontend
# serve the compiled .next directory in production mode
serve -s .next -l 3000 &
NEXT_PID=$!

# ------- 3️⃣ Wait for either process to exit -------
wait -n

# If one process fails, terminate the other and exit with its status
if kill -0 $FLASK_PID 2>/dev/null; then
  echo "🛑 Flask stopped – terminating Next.js"
  kill $NEXT_PID
fi
if kill -0 $NEXT_PID 2>/dev/null; then
  echo "🛑 Next.js stopped – terminating Flask"
  kill $FLASK_PID
fi

wait -f
