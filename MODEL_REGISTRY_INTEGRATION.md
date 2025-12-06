# Model Registry Integration Plan

## Overview

Implement a sophisticated model versioning system for a portfolio project that demonstrates practical MLOps and full-stack engineering skills. This project showcases the ability to build professional ML model management with an intuitive UI for technical demonstrations.

## Resume Value 🎯

- **MLOps Skills**: Practical model versioning, performance monitoring, A/B testing
- **Full-Stack Engineering**: React UI, FastAPI backend, PostgreSQL database design
- **System Design**: Clean architecture, API design, state management patterns
- **ML Engineering**: Model lifecycle management, benchmarking, validation

## Demo Scenarios 💡

1. **Model Upgrade Demo**: Switch from E5-base to E5-large and show performance improvement
2. **Architecture Showcase**: Explain clean service layers and API design
3. **Performance Visualization**: Demonstrate metrics comparison charts
4. **Technical Deep Dive**: Show database design and model management system

## End Goal

- ✅ Change models (text embedding, image embedding, captioning) from web UI
- ✅ See real-time performance comparisons between model versions
- ✅ Demonstrate model lifecycle management and version control
- ✅ Showcase clean architecture and engineering best practices

---

## Phase 1: Backend Model Registry Foundation

### Step 1.1: Create Model Registry Database Schema

**Files to create:**

- `backend/app/repositories/model_repository.py` - Model metadata and performance persistence

**Database tables:**

```sql
-- models table - Core model registry
CREATE TABLE models (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(100) NOT NULL,        -- e.g., "text_embedding_e5_base_v2"
    version_id VARCHAR(50) NOT NULL,       -- e.g., "v1.0.0"
    model_type VARCHAR(50) NOT NULL,       -- text_embedding, image_embedding, etc.
    source_repo VARCHAR(200) NOT NULL,     -- huggingface repo
    checkpoint_hash VARCHAR(64) NOT NULL,  -- model integrity
    inference_config JSONB NOT NULL,       -- batch_size, device, etc.
    change_history JSONB DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT FALSE,      -- Currently active version
    UNIQUE(model_id, version_id)
);

-- model_performance table - Performance tracking for comparisons
CREATE TABLE model_performance (
    id SERIAL PRIMARY KEY,
    model_type VARCHAR(50) NOT NULL,
    version_id VARCHAR(50) NOT NULL,
    test_run_id VARCHAR(100),              -- Group related test runs
    metric_name VARCHAR(100) NOT NULL,     -- latency_ms, error_rate, recall@5
    metric_value FLOAT NOT NULL,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT                            -- Test conditions, parameters, etc.
);
```

### Step 1.2: Create Model Registry Core Classes

**Files to create:**

- `backend/app/core/model_registry.py` - Model metadata management
- `backend/app/core/model_config_service.py` - Active model resolution

**Key classes:**

- `ModelVersion` - Single model version metadata
- `ModelRegistry` - Collection of all model versions
- `ModelConfigService` - Determines which model to use

### Step 1.3: Create Model Management Services

**Files to create:**

- `backend/app/services/model_management/model_switcher_service.py` - Handle model switching
- `backend/app/services/model_management/model_validator_service.py` - Validate model compatibility
- `backend/app/services/model_management/performance_tracker_service.py` - Track and compare performance

**Responsibilities:**

- Handle model switching with proper validation
- Run performance benchmarks and comparisons
- Manage model metadata and version history

---

## Phase 2: Update Existing Services for Dynamic Loading

### Step 2.1: Modify TextEmbedder Service

**File to modify:** `backend/app/services/embedding/text_embedder.py`

**Changes:**

- Accept `ModelConfigService` instead of hardcoded model name
- Implement `switch_model_version()` method for clean model switching
- Add model version metadata to embedding responses

### Step 2.2: Modify ImageEmbedder Service

**File to modify:** `backend/app/services/embedding/image_embedder.py`

**Changes:**

- Support both CLIP and SigLIP model families
- Dynamic model loading based on registry config
- Dimension compatibility checking

### Step 2.3: Modify Vision Services

**Files to modify:**

- `backend/app/services/vision/captioning_processor.py`
- `backend/app/services/vision/vision_llm_processor.py`

**Changes:**

- Registry-based model selection
- Support multiple captioning models (BLIP, BLIP-2, etc.)
- Dynamic device allocation

---

## Phase 3: Add Model Management API

### Step 3.1: Create Model API Routes

**File to create:** `backend/app/api/routes/models.py`

**Endpoints:**

```python
GET    /api/models/                              # List all registered models and versions
GET    /api/models/{model_type}                  # Get model details and current active version
POST   /api/models/{model_type}/register         # Register a new model version
POST   /api/models/{model_type}/switch/{version} # Switch active model to specific version
GET    /api/models/performance/{model_type}      # Get performance metrics for all versions
POST   /api/models/{model_type}/benchmark        # Run performance comparison test
GET    /api/models/{model_type}/compare          # Compare current vs other versions
```

### Step 3.2: Update Existing API Responses

**Files to modify:** All route files in `backend/app/api/routes/`

**Changes:**

- Add model version metadata to all responses
- Include processing latency and model info
- Enable performance tracking

### Step 3.3: Add Model Health Checks

**File to modify:** `backend/app/api/routes/health.py`

**Additions:**

- Model loading status
- Active model versions
- Model health metrics

---

## Phase 4: Frontend Model Management UI

### Step 4.1: Create Model Management Components

**Files to create:**

- `frontend/src/components/ModelManager/ModelList.tsx` - Grid of available models
- `frontend/src/components/ModelManager/ModelCard.tsx` - Individual model info card
- `frontend/src/components/ModelManager/ModelSwitcher.tsx` - Clean switch interface
- `frontend/src/components/ModelManager/ModelRegistration.tsx` - Register new models

**Features:**

- Visual model catalog with metadata
- Active version indicators
- One-click model switching
- Model registration form

### Step 4.2: Create Performance Dashboard

**Files to create:**

- `frontend/src/components/Performance/ModelComparisonChart.tsx`
- `frontend/src/components/Performance/MetricCard.tsx`
- `frontend/src/components/Performance/PerformanceDashboard.tsx`

**Features:**

- Performance comparison charts (latency, quality)
- Historical performance trends
- Benchmark test results visualization
- Side-by-side model comparisons

### Step 4.3: Add Model Context and Hooks

**Files to create:**

- `frontend/src/contexts/ModelContext.tsx`
- `frontend/src/hooks/useModelVersions.ts`
- `frontend/src/hooks/useModelPerformance.ts`

**Responsibilities:**

- Global model state management
- Real-time performance data
- Model switching coordination

### Step 4.4: Update Main Application

**File to modify:** `frontend/src/App.tsx`

**Additions:**

- Model management routes
- Performance dashboard route
- Model context provider
- Navigation to model management

---

## Phase 5: Performance Benchmarking System

### Step 5.1: Add Metrics Collection

**Files to modify:** All service classes

**Changes:**

- Track request latency per model
- Count successful vs failed operations
- Monitor model-specific metrics (embedding dimensions, etc.)
- Log performance during benchmark runs

### Step 5.2: Benchmark Test Framework

**Implementation:**

- Standardized test query sets
- Automated performance comparison
- Result persistence to database
- Export capabilities for demos

### Step 5.3: Comparison Analytics

**Features:**

- Performance improvement calculations
- Clear before/after comparisons
- Historical trend visualization
- Exportable comparison results

---

## Phase 7: Testing and Validation

### Step 7.1: Integration Tests

**Test coverage:**

- Model switching without service restart
- Performance metric collection
- Rollback functionality
- UI model management
- Multi-model concurrent operation

### Step 7.2: End-to-End Demo Testing

**Scenarios:**

- Switch text embedding models via UI
- Verify performance improvements display correctly
- Test model registration and version management
- Validate performance comparison visualizations

---

## Implementation Order (Recommended)

### Week 1: Backend Foundation

1. Database schema and model repository
2. Model registry core classes
3. Model config service

### Week 2: Service Updates

1. Update TextEmbedder for dynamic loading
2. Update ImageEmbedder for dynamic loading
3. Update vision services

### Week 3: API and Monitoring

1. Model management API endpoints
2. Performance tracking integration
3. Health check updates

### Week 4: Frontend UI

1. Model management components
2. Performance dashboard
3. Model context and hooks

### Week 5: Benchmarking & Analytics

1. Performance benchmarking system
2. Comparison analytics and visualizations
3. Demo scenario preparation

### Week 6: Testing and Demo Preparation

1. Integration testing
2. End-to-end demo validation
3. UI/UX polish and portfolio presentation setup

---

## Success Metrics

### Demo Requirements

- ✅ Change models via UI without restart
- ✅ Visual performance comparisons between versions
- ✅ <30 second model switching time
- ✅ Intuitive model registration and management
- ✅ Professional UI for technical demonstrations
- ✅ Clear performance comparison visualizations

### Technical Requirements

- ✅ Reliable model validation and compatibility checking
- ✅ Effective performance benchmarking and comparison
- ✅ Clean API design for model operations
- ✅ Proper error handling and user feedback

### Portfolio Requirements

- ✅ Demonstrates practical MLOps implementation
- ✅ Shows full-stack development capabilities
- ✅ Exhibits clean architecture and best practices
- ✅ Professional, intuitive UI design

---

## Risk Mitigation

### Technical Risks

- **Model loading failures**: Clear error messages with recovery options
- **Memory management**: Efficient model loading/unloading
- **Model compatibility**: Validation before allowing switches
- **Data integrity**: Proper database transaction handling

### Demo Risks

- **UI complexity**: Keep interface clean and user-friendly
- **Performance consistency**: Standardized test conditions
- **Loading times**: Optimize for smooth demo experience
- **Error handling**: Intuitive error states and recovery

---

## Dependencies

### Backend Dependencies

- SQLAlchemy (existing)
- FastAPI (existing)
- Pydantic (existing)
- New: `psycopg2-binary` for PostgreSQL

### Frontend Dependencies

- React (existing)
- TypeScript (existing)
- New: `recharts` for performance charts
- New: `socket.io-client` for real-time updates

### Infrastructure Dependencies

- PostgreSQL database (existing)
- Docker for containerization (existing)

---

## Future Enhancements

### Phase 7: Advanced Portfolio Features

- **Automated benchmark suites**: Pre-defined test scenarios for demos
- **Model comparison reports**: Export performance comparisons
- **Advanced visualizations**: Enhanced performance charts
- **Demo scenarios**: Pre-configured model switching demonstrations
- **Presentation mode**: Clean UI for technical presentations
