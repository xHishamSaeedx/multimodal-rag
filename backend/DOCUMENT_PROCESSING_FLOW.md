# Document Processing Flow - Mermaid Diagram

```mermaid
flowchart TD
    Start([File Upload<br/>POST /api/v1/ingest]) --> ValidateFile{Validate File}
    ValidateFile -->|Empty/Invalid| Error1[Return 400 Error]
    ValidateFile -->|Unsupported Type| Error2[Return 400 Error<br/>Unsupported file type]
    ValidateFile -->|Valid| ReadFile[Read File Bytes]
    
    ReadFile --> CheckEmpty{File Empty?}
    CheckEmpty -->|Yes| Error3[Return 400 Error<br/>File is empty]
    CheckEmpty -->|No| InitPipeline[Initialize IngestionPipeline]
    
    InitPipeline --> Step1[Step 1: Store Raw Document<br/>MinIOStorage.upload_raw_document<br/>MinIO Data Lake]
    
    Step1 --> Step2[Step 2: Extract Content<br/>ExtractionRunner.extract_parallel_from_bytes]
    
    Step2 --> ExtractText[Extract Text<br/>TextExtractor]
    Step2 --> ExtractTables[Extract Tables<br/>TableExtractor]
    Step2 --> ExtractImages[Extract Images<br/>ImageExtractor<br/>with OCR]
    
    ExtractText --> Step3[Step 3: Chunk Text<br/>TextChunker.chunk_document<br/>chunk_size: 800<br/>chunk_overlap: 150]
    ExtractTables --> Step4_5[Step 4.5: Process Tables<br/>TableProcessor.process_table]
    ExtractImages --> Step6_6[Step 6.6: Process Images]
    
    Step3 --> Step4[Step 4: Store in Database<br/>DocumentRepository.create_document<br/>DocumentRepository.create_chunks<br/>Supabase PostgreSQL]
    
    Step4_5 --> ProcessTable[Convert Table to:<br/>- JSON format<br/>- Markdown format<br/>- Flattened text]
    ProcessTable --> CreateTableChunks[Create Table Chunks<br/>One chunk per table]
    CreateTableChunks --> StoreTableChunks[Store Table Chunks<br/>DocumentRepository.create_chunks]
    StoreTableChunks --> StoreTableData[Store Table Data<br/>DocumentRepository.create_tables_batch]
    
    Step6_6 --> GenerateCaption[Generate Caption<br/>VisionProcessor.process_image<br/>Always generate captions]
    
    GenerateCaption --> UploadImage[Upload Image to Supabase<br/>SupabaseImageStorage.upload_image]
    
    UploadImage --> StoreImageRecord[Store Image Record<br/>DocumentRepository.create_image]
    StoreImageRecord --> CreateImageChunk[Create Image Chunk<br/>Generate descriptive text]
    CreateImageChunk --> StoreImageChunk[Store Image Chunk<br/>DocumentRepository.create_chunk]
    
    Step4 --> Step5[Step 5: Generate Text Embeddings<br/>TextEmbedder.embed_batch<br/>e5-base-v2 model<br/>768 dimensions]
    
    StoreTableData --> Step5_5[Step 5.5: Generate Table Embeddings<br/>TextEmbedder.embed_batch<br/>Same model as text<br/>768 dimensions]
    
    StoreImageChunk --> Step6_7[Step 6.7: Generate Image Embeddings<br/>ImageEmbedder.embed_batch<br/>CLIP model<br/>1024 dimensions]
    
    Step5 --> Step6[Step 6: Store Text Embeddings<br/>VectorRepository.store_vectors<br/>Qdrant text_chunks collection]
    
    Step5_5 --> Step6_5[Step 6.5: Store Table Embeddings<br/>VectorRepository.store_table_vectors<br/>Qdrant table_chunks collection]
    
    Step6_7 --> Step6_8[Step 6.8: Store Image Embeddings<br/>VectorRepository.store_vectors<br/>Qdrant image_chunks collection]
    
    Step6 --> Step7[Step 7: Index Text Chunks<br/>SparseRepository.index_chunks<br/>Elasticsearch BM25]
    
    Step6_5 --> Step7_5[Step 7.5: Index Table Chunks<br/>SparseRepository.index_chunks<br/>Elasticsearch BM25<br/>Using table_markdown]
    
    Step7 --> Complete[Pipeline Complete<br/>Document ID returned<br/>Chunk counts returned]
    Step7_5 --> Complete
    Step6_8 --> Complete
    
    Error1 --> End([End])
    Error2 --> End
    Error3 --> End
    Complete --> End
    
    style Start fill:#e1f5ff
    style Complete fill:#c8e6c9
    style Error1 fill:#ffcdd2
    style Error2 fill:#ffcdd2
    style Error3 fill:#ffcdd2
    style Step1 fill:#fff3e0
    style Step2 fill:#e8f5e9
    style Step5 fill:#f3e5f5
    style Step6 fill:#f3e5f5
    style Step7 fill:#f3e5f5
```

## Key Components

### 1. Ingestion Endpoint (`/api/v1/ingest`)
- **File**: `backend/app/api/routes/ingest.py`
- Accepts file uploads (PDF, DOCX, TXT, MD)
- Validates file type and size
- Orchestrates the complete ingestion pipeline

### 2. Ingestion Pipeline (`IngestionPipeline`)
- **File**: `backend/app/services/ingestion/pipeline.py`
- Orchestrates the complete document processing flow
- Coordinates extraction, chunking, embedding, and indexing

### 3. Content Extraction (`ExtractionRunner`)
- **Parallel extraction** of text, tables, and images
- Uses OCR for text extraction from images
- Extracts structured data (tables) and visual content (images)

### 4. Chunking (`TextChunker`)
- Splits text into semantic chunks
- Default: 800 tokens per chunk, 150 token overlap
- Preserves document structure

### 5. Table Processing (`TableProcessor`)
- Converts tables to multiple formats:
  - **JSON**: Structured data format
  - **Markdown**: Human-readable table format
  - **Flattened text**: For embedding generation
- Creates one chunk per table

### 6. Image Processing
- **Upload**: Stores images in Supabase Storage
- **Captioning**: Always generates captions using captioning processor (BLIP-2, etc.)
- **OCR**: Extracts text from images
- Creates descriptive chunks for images
- Note: Vision LLM mode can still be used at query time for enhanced understanding

### 7. Embedding Generation
- **Text Embeddings**: `TextEmbedder` (e5-base-v2, 768 dim)
- **Table Embeddings**: Same model as text (768 dim)
- **Image Embeddings**: `ImageEmbedder` (CLIP, 1024 dim)
- Batch processing for efficiency

### 8. Vector Storage (Qdrant)
- **text_chunks**: Text chunk embeddings
- **table_chunks**: Table chunk embeddings
- **image_chunks**: Image chunk embeddings
- Each collection stores vectors with metadata payloads

### 9. Sparse Indexing (Elasticsearch)
- **BM25 indexing** for keyword search
- Indexes text chunks and table chunks (using markdown)
- Enables hybrid search (sparse + dense)

## Data Flow

1. **Raw File** → File bytes uploaded via API
2. **MinIO Storage** → Raw document stored in data lake
3. **Extracted Content** → Text, tables, and images extracted
4. **Chunks** → Text split into chunks, tables and images processed
5. **Database** → Document and chunks stored in Supabase PostgreSQL
6. **Embeddings** → Vector representations generated
7. **Qdrant** → Embeddings stored in vector database
8. **Elasticsearch** → Chunks indexed for BM25 search
9. **Complete** → Document ready for retrieval

## Storage Systems

- **MinIO**: Raw document storage (data lake)
- **Supabase PostgreSQL**: Document and chunk metadata
- **Supabase Storage**: Image files
- **Qdrant**: Vector embeddings (3 collections)
- **Elasticsearch**: BM25 sparse index

## Processing Modes

### Vision Processing Modes:
- **captioning**: Always generates captions during ingestion (default behavior)
- **vision_llm**: Can be used at query time for enhanced image understanding (in addition to captions)

