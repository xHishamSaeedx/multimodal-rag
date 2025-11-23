# Backend Log Viewing Guide

## Quick Access

### Via Grafana (Recommended for Docker containers)

1. **Open Grafana**: http://localhost:3001
2. **Login**: `admin` / `admin`
3. **Navigate**: Go to **Explore** (compass icon in left sidebar)
4. **Select Datasource**: Choose **Loki**

### Useful LogQL Queries

```logql
# View all logs from all containers
{job="docker-containers"}

# View all logs with JSON parsing (structured logs)
{job="docker-containers"} | json

# Filter by container name
{container_name="multimodal-rag-qdrant"}

# Filter by service name (if backend runs in Docker)
{service="backend"}

# Filter by log level
{job="docker-containers"} | json | level="ERROR"
{job="docker-containers"} | json | level="INFO"

# Search for specific text
{job="docker-containers"} |= "error"
{job="docker-containers"} |= "request"

# Filter by correlation ID (for request tracing)
{job="docker-containers"} | json | correlation_id="abc-123"

# Count errors per minute
rate({job="docker-containers"} | json | level="ERROR" [1m])

# View logs from last 5 minutes
{job="docker-containers"} | json
```

## Option 2: Direct Docker Logs (If Backend Runs in Docker)

If your backend runs as a Docker container:

```bash
# View all backend logs
docker logs multimodal-rag-backend

# Follow logs in real-time
docker logs -f multimodal-rag-backend

# View last 100 lines
docker logs --tail 100 multimodal-rag-backend

# View logs with timestamps
docker logs -t multimodal-rag-backend
```

## Option 3: Local Backend (Running Outside Docker)

If your backend runs locally (not in Docker), you have two options:

### A. View logs in terminal
The backend outputs JSON logs to stdout. Just check your terminal where you started the backend.

### B. Set up file-based log collection

1. **Configure backend to write logs to file** (modify `backend/app/utils/logging.py` if needed)

2. **Uncomment file-based scraping in Promtail**:
   Edit `backend/observability/promtail/promtail-config.yml` and uncomment the `application-logs` job (lines 101-134)

3. **Mount log directory** in docker-compose.yml:
   ```yaml
   promtail:
     volumes:
       - ./backend/observability/promtail/promtail-config.yml:/etc/promtail/config.yml
       - /var/lib/docker/containers:/var/lib/docker/containers:ro
       - /var/run/docker.sock:/var/run/docker.sock:ro
       - ./logs:/var/log/app:ro  # Add this line
   ```

4. **Restart Promtail**:
   ```bash
   docker-compose restart promtail
   ```

## Option 4: Run Backend in Docker

To have Promtail automatically collect backend logs, run the backend in Docker:

1. **Add backend service to docker-compose.yml**:
   ```yaml
   backend:
     build: ./backend
     container_name: multimodal-rag-backend
     ports:
       - "8000:8000"
     environment:
       - DEBUG=true
     volumes:
       - ./backend:/app
     networks:
       - rag-network
     depends_on:
       - qdrant
       - elasticsearch
       - minio
   ```

2. **Start backend**:
   ```bash
   docker-compose up -d backend
   ```

3. **View logs in Grafana** using:
   ```logql
   {service="backend"} | json
   ```

## Quick Commands Reference

```bash
# Check Promtail is collecting logs
docker logs multimodal-rag-promtail

# Check Loki is receiving logs
docker logs multimodal-rag-loki

# Test Loki API directly
curl http://localhost:3100/ready

# View all containers on rag-network
docker ps --filter "network=multimodal-rag_rag-network"
```

## Troubleshooting

### No logs appearing in Grafana?

1. **Check Promtail is running**:
   ```bash
   docker ps | grep promtail
   ```

2. **Check Promtail logs**:
   ```bash
   docker logs multimodal-rag-promtail
   ```

3. **Verify backend container is on rag-network**:
   ```bash
   docker inspect <backend-container-name> | grep NetworkMode
   ```

4. **Check if backend is outputting logs**:
   ```bash
   docker logs <backend-container-name>
   ```

### Backend logs not structured JSON?

Make sure your backend is using the logging configuration from `backend/app/utils/logging.py` with `json_output=True`.

