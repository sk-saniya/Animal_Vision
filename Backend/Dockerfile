# ----------- Build the Next.js frontend (static assets) -----------
FROM node:20-alpine AS builder
WORKDIR /app

# Install frontend dependencies & build
COPY frontend/package*.json ./frontend/
RUN cd frontend && npm ci
COPY frontend ./frontend
RUN cd frontend && npm run build   # creates .next static output

# ----------- Runtime image (Python + Node) ------------------------
FROM python:3.11-slim

# Install backend Python dependencies
WORKDIR /app
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r ./backend/requirements.txt

# Copy backend source
COPY backend ./backend

# Copy the built Next.js assets from the builder stage
COPY --from=builder /app/frontend/.next ./frontend/.next
COPY --from=builder /app/frontend/public ./frontend/public
COPY --from=builder /app/frontend/next.config.js ./frontend/
COPY --from=builder /app/frontend/package.json ./frontend/

# Install a small static‑file server for the built Next.js app
RUN npm install -g serve@14

# Expose the ports we will use
EXPOSE 3000 5000

# Copy the start‑up script and make it executable
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Environment variable used by the frontend to reach the backend
# (Railway will replace the value with its own if you set it in the UI)
ENV FLASK_URL=http://localhost:5000

# Run both processes
CMD ["/start.sh"]



FROM python:3.9-slim # Install system dependencies for CLIP RUN apt-get update && apt-get install -y \ git libgl1-mesa-glx libglib2.0-0 \ && rm -rf /var/lib/apt/lists/* WORKDIR /app # Copy and install Python dependencies COPY requirements.txt . RUN pip install --no-cache-dir -r requirements.txt # Copy all backend files COPY . . # HF Spaces requires port 7860 EXPOSE 7860 # Start Flask on port 7860 CMD ["python", "clip_api.py"]