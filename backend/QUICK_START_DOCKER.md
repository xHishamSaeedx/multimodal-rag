# Quick Start: Backend in Docker

## 🚀 One-Time Setup

```bash
# Build and start backend (with all dependencies)
docker-compose up -d backend
```

## 💻 Daily Development Workflow

### 1. Start Backend (if not running)
```bash
docker-compose up -d backend
```

### 2. Code & Edit
- **Just edit your code files** in `backend/`
- **Save the file** - uvicorn automatically detects changes
- **Wait 1-2 seconds** - backend restarts automatically
- **Test your changes** - no manual restart needed!

### 3. View Logs
```bash
# Option 1: Docker logs (quick)
docker logs -f multimodal-rag-backend

# Option 2: Grafana (recommended - better search)
# Open http://localhost:3001
# Explore → Loki → Query: {service="backend"} | json
```

### 4. Stop Backend (when done)
```bash
docker-compose stop backend
```

## 🔄 Comparison: Local vs Docker

| What You Do | Local | Docker |
|------------|-------|--------|
| **Start** | `uvicorn main:app --reload` | `docker-compose up -d backend` |
| **Edit Code** | Save file → auto-reload | Save file → auto-reload ✅ |
| **See Changes** | Instant | Instant (1-2 sec) ✅ |
| **View Logs** | Terminal | `docker logs -f` or Grafana |
| **Stop** | Ctrl+C | `docker-compose stop backend` |

## 🎯 Key Points

✅ **Hot-reload works!** - Code changes are automatically detected  
✅ **No manual restart needed** - Just save and wait 1-2 seconds  
✅ **Logs in Grafana** - Better than terminal for searching  
✅ **Isolated environment** - Won't mess with your system Python  

## 📝 Common Commands

```bash
# Start
docker-compose up -d backend

# View logs
docker logs -f multimodal-rag-backend

# Restart (if needed)
docker-compose restart backend

# Stop
docker-compose stop backend

# Rebuild (after changing requirements.txt)
docker-compose up -d --build backend
```

## 🔍 Troubleshooting

**Changes not showing?**
- Wait 2-3 seconds (uvicorn needs time to detect)
- Check logs: `docker logs multimodal-rag-backend`
- Restart: `docker-compose restart backend`

**Can't connect to services?**
- Check services are running: `docker-compose ps`
- Backend uses service names: `qdrant`, `elasticsearch`, `minio` (not localhost)

**Port 8000 in use?**
- Change port in docker-compose.yml: `"8001:8000"`

## 📚 More Info

See `DOCKER_DEVELOPMENT.md` for detailed guide.

