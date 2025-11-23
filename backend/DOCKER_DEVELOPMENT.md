# Backend Docker Development Guide

## Overview

This guide explains how to develop the backend in Docker with **hot-reload** enabled, so code changes are automatically reflected without restarting the container.

## Architecture

```
Your Code Changes (on host)
    ↓
Docker Volume Mount (./backend → /app in container)
    ↓
Uvicorn with --reload flag (watches for file changes)
    ↓
Auto-restarts Python process (fast, no container restart needed)
```

## Quick Start

### 1. Start Backend in Docker

```bash
# Start backend along with all dependencies
docker-compose up -d backend

# Or start everything
docker-compose up -d
```

### 2. Verify It's Running

```bash
# Check container status
docker ps | grep backend

# View logs
docker logs -f multimodal-rag-backend

# Test the API
curl http://localhost:8000/
```

### 3. Start Coding!

**Just edit your code files** - changes are automatically detected and the server restarts within seconds!

## Development Workflow

### Local Development (Before Docker)

**Old way (local):**

```bash
# Terminal 1: Start backend
cd backend
uvicorn main:app --reload

# Make code changes...

# Terminal 2: Stop backend (Ctrl+C)
# Terminal 1: Start again
uvicorn main:app --reload
```

### Docker Development (New Way)

**New way (Docker with hot-reload):**

```bash
# Terminal 1: Start backend in Docker (one time)
docker-compose up -d backend

# Make code changes in your editor...

# That's it! Changes are automatically detected and applied!
# No need to stop/restart anything!
```

## How Hot-Reload Works

1. **Volume Mount**: Your `./backend` directory is mounted to `/app` in the container
2. **File Watching**: Uvicorn's `--reload` flag watches for file changes
3. **Auto-Restart**: When you save a file, uvicorn detects the change and restarts the Python process
4. **Fast**: Only the Python process restarts (1-2 seconds), not the entire container

## Common Development Tasks

### View Logs

```bash
# Follow logs in real-time
docker logs -f multimodal-rag-backend

# View last 50 lines
docker logs --tail 50 multimodal-rag-backend

# View logs with timestamps
docker logs -t multimodal-rag-backend
```

### View Logs in Grafana (Recommended)

1. Open http://localhost:3001
2. Login: `admin` / `admin`
3. Go to **Explore** → Select **Loki**
4. Query: `{service="backend"} | json`

### Restart Backend (if needed)

```bash
# Restart just the backend
docker-compose restart backend

# Rebuild and restart (if you change dependencies)
docker-compose up -d --build backend
```

### Stop Backend

```bash
# Stop backend
docker-compose stop backend

# Stop and remove container
docker-compose down backend
```

### Install New Python Package

1. **Add to requirements.txt**:

   ```bash
   echo "new-package>=1.0.0" >> backend/requirements.txt
   ```

2. **Rebuild container**:

   ```bash
   docker-compose up -d --build backend
   ```

3. **Or install in running container** (temporary):
   ```bash
   docker exec -it multimodal-rag-backend pip install new-package
   ```
   (Note: This won't persist - add to requirements.txt for permanent install)

### Access Container Shell

```bash
# Open interactive shell in backend container
docker exec -it multimodal-rag-backend bash

# Run Python commands
docker exec -it multimodal-rag-backend python -c "import sys; print(sys.version)"

# Run tests
docker exec -it multimodal-rag-backend pytest
```

### Debug Issues

```bash
# Check if backend is running
docker ps | grep backend

# Check container logs for errors
docker logs multimodal-rag-backend

# Check if backend can reach other services
docker exec -it multimodal-rag-backend curl http://qdrant:6333/health
docker exec -it multimodal-rag-backend curl http://elasticsearch:9200/_cluster/health

# Inspect container
docker inspect multimodal-rag-backend
```

## Environment Variables

The backend uses environment variables for configuration. In Docker, these are set in `docker-compose.yml`:

```yaml
environment:
  - QDRANT_HOST=qdrant # Docker service name
  - ELASTICSEARCH_URL=http://elasticsearch:9200
  - MINIO_ENDPOINT=minio:9000
  - DEBUG=true
```

### Override Environment Variables

**Option 1: Edit docker-compose.yml** (recommended for team)

**Option 2: Use .env file** (for local overrides)

```bash
# Create backend/.env
QDRANT_HOST=qdrant
ELASTICSEARCH_URL=http://elasticsearch:9200
DEBUG=true
```

**Option 3: Pass at runtime**

```bash
docker-compose run -e DEBUG=false backend
```

## Network Configuration

The backend is on the `rag-network` Docker network, which allows it to:

- Connect to other services using service names:
  - `qdrant` (instead of `localhost`)
  - `elasticsearch` (instead of `localhost`)
  - `minio` (instead of `localhost`)
- Have its logs automatically collected by Promtail
- Be accessible from host at `http://localhost:8000`

## Performance Tips

### Faster Hot-Reload

- **Exclude large directories** in `.dockerignore` (already done)
- **Don't mount node_modules** or other build artifacts
- **Use volume exclusions** in docker-compose.yml (already configured)

### Reduce Container Size

The Dockerfile uses `python:3.11-slim` for a smaller image. If you need more tools, you can switch to `python:3.11` but it will be larger.

## Troubleshooting

### Changes Not Reflecting?

1. **Check volume mount**:

   ```bash
   docker inspect multimodal-rag-backend | grep -A 10 Mounts
   ```

2. **Check uvicorn is watching**:

   ```bash
   docker logs multimodal-rag-backend | grep -i reload
   ```

3. **Manually trigger reload**:
   ```bash
   docker-compose restart backend
   ```

### Backend Can't Connect to Services?

1. **Check network**:

   ```bash
   docker network inspect multimodal-rag_rag-network
   ```

2. **Test connectivity**:

   ```bash
   docker exec -it multimodal-rag-backend ping qdrant
   ```

3. **Check service names match**:
   - In docker-compose.yml: `qdrant`, `elasticsearch`, `minio`
   - In backend config: Use these exact names

### Port Already in Use?

If port 8000 is already in use:

```bash
# Find what's using it
netstat -ano | findstr :8000  # Windows
lsof -i :8000                 # Linux/Mac

# Or change port in docker-compose.yml
ports:
  - "8001:8000"  # Use 8001 on host
```

## Comparison: Local vs Docker

| Task                | Local Development           | Docker Development                  |
| ------------------- | --------------------------- | ----------------------------------- |
| **Start**           | `uvicorn main:app --reload` | `docker-compose up -d backend`      |
| **Code Changes**    | Auto-reloads                | Auto-reloads (via volume mount)     |
| **View Logs**       | Terminal output             | `docker logs -f backend` or Grafana |
| **Install Package** | `pip install`               | Add to requirements.txt + rebuild   |
| **Stop**            | Ctrl+C                      | `docker-compose stop backend`       |
| **Restart**         | Stop + Start                | `docker-compose restart backend`    |
| **Isolation**       | Uses system Python          | Isolated environment                |
| **Dependencies**    | Must install locally        | Managed in container                |

## Best Practices

1. **Always use volume mounts** for code (already configured)
2. **Add new packages to requirements.txt** before using them
3. **Use Grafana for log viewing** instead of `docker logs` for better search
4. **Keep .dockerignore updated** to exclude unnecessary files
5. **Test locally first** for quick iterations, then verify in Docker

## Next Steps

- View logs in Grafana: http://localhost:3001
- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health
