# Service URLs

This document lists all service URLs accessible when running the multimodal-rag system via Docker Compose.

## Core Services

### Backend API (FastAPI)

- **Main API**: http://localhost:8000
- **API Documentation (Swagger)**: http://localhost:8000/docs
- **Alternative API Docs (ReDoc)**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

### Qdrant (Vector Database)

- **REST API**: http://localhost:6333 (HTTP, fallback for compatibility)
- **Dashboard/Web UI**: http://localhost:6333/dashboard
- **Health Check**: http://localhost:6333/health
- **gRPC Endpoint**: `localhost:6334` (preferred for vector operations - lower latency)
  - **Note**: The backend now uses gRPC by default for better performance (20-50ms faster)

### Elasticsearch (BM25 Sparse Index)

- **HTTP API**: http://localhost:9200
- **Cluster Health**: http://localhost:9200/\_cluster/health
- **Index Info**: http://localhost:9200/chunks
- **Transport Endpoint**: `localhost:9300` (not HTTP, for transport protocol)
- **Cleanup Script**: `python scripts/clean_elasticsearch.py --help` (for removing documents)

### MinIO (S3-compatible Storage)

- **S3 API**: http://localhost:9000
- **Console UI**: http://localhost:9090
  - **Default Username**: `admin`
  - **Default Password**: `admin12345`
- **Health Check**: http://localhost:9000/minio/health/live

## Observability Services

### Grafana (Visualization & Dashboards)

- **Web UI**: http://localhost:3001
  - **Default Username**: `admin`
  - **Default Password**: `admin`
- **Health Check**: http://localhost:3001/api/health

**Note**: Grafana uses port 3001 to avoid conflicts with frontend applications (typically on port 3000).

### Grafana Loki (Log Aggregation)

- **HTTP API**: http://localhost:3100
- **Readiness Check**: http://localhost:3100/ready
- **Metrics**: http://localhost:3100/metrics

### Promtail (Log Collection Agent)

- **HTTP API**: http://localhost:9080
- **Metrics**: http://localhost:9080/metrics

## Quick Access Links

### Most Frequently Used

- **Grafana Dashboard**: http://localhost:3001 (login: admin/admin)
- **Qdrant Dashboard**: http://localhost:6333/dashboard
- **MinIO Console**: http://localhost:9090 (login: admin/admin12345)
- **API Documentation**: http://localhost:8000/docs
- **Elasticsearch**: http://localhost:9200

## Network Notes

All services run on the `rag-network` Docker network and can communicate using their service names:

- `qdrant` (instead of localhost)
- `elasticsearch` (instead of localhost)
- `minio` (instead of localhost)
- `backend` (instead of localhost)

When accessing from **outside Docker** (your browser or local machine), use `localhost` with the ports listed above.

When accessing from **inside Docker containers** (service-to-service communication), use the service names (e.g., `http://qdrant:6333`).

## Service Status Checks

### Verify All Services Are Running

```bash
# Check Docker containers
docker ps

# Check specific services
curl http://localhost:6333/health          # Qdrant
curl http://localhost:9200/_cluster/health # Elasticsearch
curl http://localhost:8000/health          # Backend API
curl http://localhost:9000/minio/health/live # MinIO
curl http://localhost:3100/ready            # Loki
curl http://localhost:3001/api/health       # Grafana
```

## Default Credentials

| Service       | Username | Password     | Notes                  |
| ------------- | -------- | ------------ | ---------------------- |
| MinIO Console | `admin`  | `admin12345` | Change in production!  |
| Grafana       | `admin`  | `admin`      | Change on first login! |

**⚠️ Security Warning**: These are default development credentials. **Always change them in production environments!**
