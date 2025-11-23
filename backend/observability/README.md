# Observability Stack - Grafana Loki + Promtail

This directory contains the configuration for centralized log aggregation using Grafana Loki and Promtail.

## Overview

The observability stack consists of:

- **Grafana Loki**: Log aggregation system that collects and stores logs
- **Promtail**: Log collection agent that scrapes logs from containers and sends them to Loki
- **Grafana**: Visualization platform for querying and visualizing logs

## Architecture

```
FastAPI Backend (stdout JSON logs)
    ↓
Promtail (collects from Docker containers)
    ↓
Grafana Loki (stores logs)
    ↓
Grafana (query and visualize)
```

## Directory Structure

```
observability/
├── loki/
│   └── loki-config.yml          # Loki server configuration
├── promtail/
│   └── promtail-config.yml      # Promtail log collection configuration
├── grafana/
│   └── provisioning/
│       ├── datasources/
│       │   └── loki.yml         # Auto-configure Loki datasource
│       └── dashboards/
│           └── dashboard.yml    # Dashboard provisioning config
└── README.md                     # This file
```

## Quick Start

### 1. Start the Observability Stack

The observability services are included in the main `docker-compose.yml` file. Start them along with other services:

```bash
# Start all services including Loki, Promtail, and Grafana
docker-compose up -d

# Or start only observability services
docker-compose up -d loki promtail grafana
```

### 2. Access Grafana

- **URL**: http://localhost:3001
- **Username**: `admin`
- **Password**: `admin`

Loki is automatically configured as a datasource in Grafana.

### 3. View Logs

Once Grafana is running:

1. Open Grafana at http://localhost:3001
2. Go to **Explore** (compass icon in the left sidebar)
3. Select **Loki** as the datasource
4. Use LogQL queries to search logs

## LogQL Query Examples

### Basic Queries

```logql
# View all logs
{job="docker-containers"}

# Filter by container name
{container_name="multimodal-rag-qdrant"}

# Filter by service name
{service="backend"}

# Filter by log level
{job="docker-containers"} |= "ERROR"

# Filter by correlation ID
{job="docker-containers"} | json | correlation_id="abc-123"
```

### Advanced Queries

```logql
# Filter by log level and parse JSON
{job="docker-containers"} | json | level="ERROR"

# Filter by event type
{job="docker-containers"} | json | event="request_error"

# Count logs by level
sum by (level) (count_over_time({job="docker-containers"} | json [5m]))

# Count errors per minute
rate({job="docker-containers"} | json | level="ERROR" [1m])
```

### Query by Correlation ID

To trace a request end-to-end using correlation IDs:

```logql
{job="docker-containers"} | json | correlation_id="your-correlation-id-here"
```

The correlation ID is included in response headers as `X-Correlation-ID`.

## Configuration Details

### Loki Configuration (`loki/loki-config.yml`)

- **Port**: 3100 (HTTP API)
- **Storage**: Filesystem-based storage in Docker volume
- **Retention**: 7 days (configurable in `limits_config`)
- **Query Limits**: 50,000 entries per query

Key settings:

- `auth_enabled: false` - No authentication (for development)
- `replication_factor: 1` - Single instance setup
- `max_entries_limit_per_query: 50000` - Query result limit

### Promtail Configuration (`promtail/promtail-config.yml`)

Promtail automatically discovers and scrapes logs from Docker containers on the `rag-network`.

**Features:**

- Auto-discovers containers via Docker socket
- Parses JSON logs from structlog
- Extracts labels: `container_name`, `service`, `level`, `logger`, `event`, `correlation_id`
- Handles Docker JSON log format

**Pipeline Stages:**

1. Parses Docker JSON log format
2. Extracts application JSON logs (from structlog)
3. Adds labels from parsed JSON fields
4. Formats output for Loki

### Grafana Provisioning

The Grafana configuration automatically:

- Sets up Loki as a datasource
- Configures default settings
- Enables dashboard provisioning

## Backend Application Integration

The FastAPI backend is already configured for structured logging:

- **Logging Library**: `structlog` (already in requirements.txt)
- **Output Format**: JSON to stdout (container-friendly)
- **Correlation IDs**: Automatically added via middleware
- **Log Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL

The backend outputs JSON logs to stdout, which Docker captures. Promtail then collects these logs and sends them to Loki.

### Viewing Backend Logs

If your backend runs as a Docker container on the `rag-network`, Promtail will automatically collect its logs. If running locally, you can:

1. **Option 1**: Run backend in Docker with proper networking
2. **Option 2**: Use file-based log collection (uncomment in promtail-config.yml)

## Ports

| Service  | Port | Purpose                   |
| -------- | ---- | ------------------------- |
| Loki     | 3100 | HTTP API                  |
| Promtail | 9080 | HTTP API (status/metrics) |
| Grafana  | 3001 | Web UI                    |

**Note**: Grafana uses port 3001 to avoid conflicts with the frontend (typically on 3000).

## Docker Volumes

The following volumes are created:

- `loki_data`: Loki storage (logs and indexes)
- `promtail_positions`: Promtail position tracking
- `grafana_data`: Grafana data and dashboards

## Monitoring Log Collection

### Check Promtail Status

```bash
# View Promtail logs
docker logs multimodal-rag-promtail

# Check Promtail metrics
curl http://localhost:9080/metrics
```

### Check Loki Status

```bash
# View Loki logs
docker logs multimodal-rag-loki

# Check Loki readiness
curl http://localhost:3100/ready

# Check Loki metrics
curl http://localhost:3100/metrics
```

### Check Grafana Status

```bash
# View Grafana logs
docker logs multimodal-rag-grafana

# Check Grafana health
curl http://localhost:3001/api/health
```

## Log Retention

By default, logs are retained for **7 days** (168 hours). You can modify this in `loki/loki-config.yml`:

```yaml
limits_config:
  reject_old_samples_max_age: 168h # Change this value
```

## Troubleshooting

### Grafana Cannot Connect to Loki (DNS Resolution Error)

If you see errors like `dial tcp: lookup loki on 127.0.0.11:53: no such host`:

1. **Restart all services to ensure proper network initialization**:

   ```bash
   docker-compose down
   docker-compose up -d loki promtail grafana
   ```

2. **Verify containers are on the same network**:

   ```bash
   docker network inspect rag-network
   ```

   You should see `multimodal-rag-loki`, `multimodal-rag-promtail`, and `multimodal-rag-grafana` listed.

3. **Test DNS resolution from Grafana container**:

   ```bash
   docker exec multimodal-rag-grafana nslookup loki
   # Or test connectivity
   docker exec multimodal-rag-grafana wget -O- http://loki:3100/ready
   ```

4. **If DNS still fails, manually configure the datasource in Grafana**:

   - Go to http://localhost:3001
   - Navigate to **Configuration** → **Data Sources**
   - Add Loki datasource with URL: `http://loki:3100`
   - Or use container name: `http://multimodal-rag-loki:3100`
   - Save and Test

5. **Alternative: Use host network mode (not recommended for production)**:
   If DNS continues to fail, you can use `host.docker.internal` or the container's IP:

   ```bash
   # Get Loki container IP
   docker inspect multimodal-rag-loki | grep IPAddress
   ```

   Then update the datasource URL to use the IP address.

6. **Check Docker Desktop network settings (Windows/Mac)**:
   - Ensure Docker Desktop is using the default bridge network
   - Try restarting Docker Desktop
   - Check if VPN or firewall is interfering with Docker networking

### Logs Not Appearing in Grafana

1. **Check Promtail is running**:

   ```bash
   docker ps | grep promtail
   ```

2. **Check Promtail is discovering containers**:

   ```bash
   docker logs multimodal-rag-promtail
   ```

3. **Verify container is on the rag-network**:

   ```bash
   docker network inspect rag-network
   ```

4. **Check Loki is receiving logs**:
   ```bash
   curl http://localhost:3100/ready
   curl http://localhost:3100/loki/api/v1/label/container_name/values
   ```

### Backend Logs Not Being Collected

If your backend runs outside Docker:

1. **Option 1**: Add the backend container to docker-compose.yml
2. **Option 2**: Configure file-based collection in Promtail (see commented section)

### Performance Issues

- **Reduce log retention**: Lower `reject_old_samples_max_age` in loki-config.yml
- **Increase query limits**: Adjust `max_entries_limit_per_query`
- **Optimize LogQL queries**: Use label filters before parsing JSON

## Production Considerations

For production deployments:

1. **Enable Authentication**: Set `auth_enabled: true` in Loki and configure authentication
2. **Use Object Storage**: Configure S3/GCS for Loki storage instead of filesystem
3. **Horizontal Scaling**: Deploy multiple Loki instances with replication
4. **Monitoring**: Monitor Loki and Promtail metrics with Prometheus
5. **Backup**: Regularly backup Grafana dashboards and Loki data
6. **Security**: Use HTTPS and secure credentials

## References

- [Grafana Loki Documentation](https://grafana.com/docs/loki/latest/)
- [Promtail Documentation](https://grafana.com/docs/loki/latest/clients/promtail/)
- [LogQL Query Language](https://grafana.com/docs/loki/latest/logql/)
- [Grafana Documentation](https://grafana.com/docs/grafana/latest/)

## Related Documentation

- See `logging.md` in the project root for logging architecture and best practices
- See `docs/phase-1-foundation.md` for Phase 1 implementation details
