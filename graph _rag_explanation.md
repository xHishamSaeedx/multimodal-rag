I'll create a comprehensive documentation of the GraphRAG system. Let me first read the key implementation files to capture all the details.

[7 tools called]

Now I'll create a comprehensive markdown document describing the entire GraphRAG system. Since you're in **ask mode**, I'll provide the content for you to save.

---

# **GraphRAG System Architecture - Complete End-to-End Documentation**

## **📋 Table of Contents**

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Document Ingestion Pipeline](#document-ingestion-pipeline)
4. [Graph Schema & Data Model](#graph-schema--data-model)
5. [Entity & Topic Extraction](#entity--topic-extraction)
6. [Graph Construction](#graph-construction)
7. [Query Processing & Retrieval](#query-processing--retrieval)
8. [Hybrid Retrieval Integration](#hybrid-retrieval-integration)
9. [Performance & Optimization](#performance--optimization)

---

## **Overview**

This GraphRAG (Graph-based Retrieval Augmented Generation) system uses a **knowledge graph** stored in **Neo4j** to enhance document retrieval for LLM-based question answering. Unlike traditional RAG systems that rely solely on vector similarity or keyword matching, GraphRAG leverages **structured relationships** between documents, sections, chunks, entities, and topics to provide **context-aware, multi-hop retrieval**.

### **Key Design Principles**

1. **Universal Design**: Works across any domain using spaCy NER (not domain-specific ontologies)
2. **Structure-Based Retrieval**: Leverages document hierarchy (Document → Section → Chunk)
3. **Cross-Document Linking**: Entities and topics create bridges across documents
4. **Multi-Strategy Retrieval**: Combines section-aware, entity-based, topic-based, and content search
5. **Context Expansion**: Uses sequential chunk links (`NEXT_CHUNK`) for better context

---

## **System Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                    RAW DOCUMENT INPUT                       │
│          (PDF, DOCX, TXT, MD, etc.)                        │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│              EXTRACTION PIPELINE                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  1. Text Extraction (PyMuPDF, python-docx, etc.)     │  │
│  │  2. Table Detection & Extraction                     │  │
│  │  3. Image Extraction & OCR                           │  │
│  │  4. Metadata Extraction                              │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│             CHUNKING & SECTION DETECTION                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  1. Smart Section Detection (heading patterns)       │  │
│  │     - Regex patterns: "Chapter X", "#.# Title"       │  │
│  │     - Spacing normalization: "L I N K" → "LINK"      │  │
│  │     - Multi-line heading detection                   │  │
│  │  2. Text Chunking (800 tokens, 150 overlap)          │  │
│  │  3. Table Chunking (preserve structure)              │  │
│  │  4. Image Chunking (with captions)                   │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│         ENTITY & TOPIC EXTRACTION (NLP)                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  1. spaCy NER (en_core_web_sm)                       │  │
│  │     - Entities: PERSON, ORG, GPE, DATE, MONEY,       │  │
│  │       PERCENT, PRODUCT, EVENT, LAW, WORK_OF_ART      │  │
│  │     - Confidence scoring                             │  │
│  │     - Name normalization (lowercase, trim)           │  │
│  │  2. Topic Extraction                                 │  │
│  │     - From section titles                            │  │
│  │     - From entity clusters                           │  │
│  │     - Keyword extraction                             │  │
│  │  3. Entity Relationship Extraction                   │  │
│  │     - Co-occurrence analysis (same chunk)            │  │
│  │     - Frequency-based weighting                      │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│            GRAPH CONSTRUCTION (Neo4j)                       │
│                                                             │
│    Document ──HAS_SECTION──> Section ──HAS_CHUNK──> Chunk  │
│                                 │                      │    │
│                                 │                      │    │
│                            ABOUT│                MENTIONS   │
│                                 │                      │    │
│                                 ▼                      ▼    │
│                              Entity <──RELATED_TO──> Entity │
│                                 │                           │
│                          ASSOCIATED_WITH                    │
│                                 │                           │
│                                 ▼                           │
│                               Topic <──HAS_TOPIC── Chunk    │
│                                                             │
│              Chunk ──NEXT_CHUNK──> Chunk (sequential)       │
│              Chunk ──HAS_MEDIA──> Media                     │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│         PARALLEL STORAGE (Multi-Index Strategy)             │
│  ┌──────────┬──────────┬──────────┬────────────────────┐   │
│  │  Neo4j   │  Qdrant  │   Elastic│    Supabase        │   │
│  │  Graph   │  Vector  │   BM25   │    Metadata        │   │
│  │  DB      │  Store   │   Index  │    PostgreSQL      │   │
│  └──────────┴──────────┴──────────┴────────────────────┘   │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│              QUERY TIME - HYBRID RETRIEVAL                  │
│                                                             │
│  User Query: "Summarize utilitarianism in business ethics" │
│                           │                                 │
│           ┌───────────────┼───────────────┐                │
│           │               │               │                │
│           ▼               ▼               ▼                │
│      ┌────────┐     ┌─────────┐    ┌──────────┐           │
│      │ BM25   │     │ Vector  │    │  Graph   │           │
│      │ Sparse │     │ Dense   │    │ Retrieval│           │
│      │Search  │     │ Search  │    │ (Neo4j)  │           │
│      └────┬───┘     └────┬────┘    └────┬─────┘           │
│           │              │              │                  │
│           └──────────────┼──────────────┘                  │
│                          │                                 │
│                          ▼                                 │
│               ┌──────────────────────┐                     │
│               │  Score Normalization │                     │
│               │  & Result Merging    │                     │
│               └──────────┬───────────┘                     │
│                          │                                 │
│                          ▼                                 │
│                 Top-K Chunks (10-50)                       │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│              LLM ANSWER GENERATION                          │
│  Retrieved Context + Query → LLM → Final Answer            │
└─────────────────────────────────────────────────────────────┘
```

---

## **Document Ingestion Pipeline**

### **Step 1: Text Extraction**

**File**: `backend/app/services/ingestion/pipeline.py`

**Purpose**: Extract raw text, tables, and images from various document formats.

**Process**:

```python
# Extraction uses format-specific libraries
- PDF: PyMuPDF (fitz)
- DOCX: python-docx
- TXT/MD: Direct reading
- Images in PDFs: OCR with pytesseract (optional)
```

**Output**: `ExtractedContent` object containing:

- `text`: Full document text
- `tables`: List of detected tables
- `images`: List of extracted images with metadata
- `metadata`: Document-level metadata (page count, author, etc.)

---

### **Step 2: Section Detection**

**File**: `backend/app/services/ingestion/pipeline.py` → `_build_document_graph()`

**Purpose**: Identify logical sections (chapters, headings) to create meaningful document structure.

**Algorithm**:

```python
def detect_sections(text):
    sections = []
    lines = text.split('\n')

    # Heading patterns (regex)
    patterns = [
        r'^(Chapter|CHAPTER)\s+\d+',              # "Chapter 1"
        r'^\d+\.\d+\s+[A-Z]',                     # "1.1 Introduction"
        r'^[A-Z][A-Z\s]{10,50}$',                 # "INTRODUCTION TO ETHICS"
        r'^\d+\.\s+[A-Z]',                        # "1. Overview"
    ]

    for i, line in enumerate(lines):
        # Normalize spacing: "L I N K" → "LINK"
        normalized = re.sub(r'\s+', '', line) if has_excessive_spacing(line) else line

        # Check if line matches heading pattern
        for pattern in patterns:
            if re.match(pattern, normalized.strip()):
                sections.append({
                    'title': normalized.strip(),
                    'start_line': i,
                })
                break

    return sections
```

**Key Features**:

- **Spacing normalization**: PDF extraction sometimes adds spaces between letters ("L I N K" → "LINK")
- **Multi-line headers**: Checks multiple consecutive lines for potential headings
- **Fallback**: If no sections detected, creates a single "Document Content" section

**Issues & Fixes**:

- ⚠️ **Problem**: Too aggressive heading detection treats answers ("11. True") and footnotes as sections
- ✅ **Solution** (suggested, not yet implemented): Filter out lines that:
  - End with periods in numbered lines (likely answers/footnotes)
  - Are >100 characters (likely body text, not headings)
  - Don't start with proper heading patterns

---

### **Step 3: Text Chunking**

**File**: `backend/app/services/ingestion/chunker.py`

**Strategy**: Recursive character splitting with overlap

**Parameters**:

- `chunk_size`: 800 tokens (~3200 characters)
- `chunk_overlap`: 150 tokens (~600 characters)
- Preserves sentence boundaries when possible

**Example**:

```
Original Text (3000 chars):
"Ethics is the study of right and wrong. It includes utilitarianism,
which focuses on outcomes, deontology, which focuses on duties..."

Chunks:
[Chunk 1] chars 0-800: "Ethics is the study...utilitarianism..."
[Chunk 2] chars 650-1450: "...utilitarianism, which...deontology..."  (overlap)
[Chunk 3] chars 1300-2100: "...deontology, which...virtue ethics..."
```

---

### **Step 4: Entity Extraction (spaCy NER)**

**File**: `backend/app/services/ingestion/pipeline.py` → `_extract_entities_spacy()`

**Purpose**: Identify and extract named entities from text chunks using spaCy.

**Implementation**:

```python
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_entities(chunks):
    entities = []

    for chunk in chunks:
        doc = nlp(chunk.text)

        for ent in doc.ents:
            entities.append({
                "entity_id": str(uuid.uuid4()),
                "entity_name": ent.text,
                "normalized_name": ent.text.lower().strip(),  # For cross-doc matching
                "entity_type": ent.label_,  # PERSON, ORG, GPE, etc.
                "confidence": 1.0,
                "chunk_ids": [chunk.chunk_id],
            })

    # Merge entities with same normalized_name
    merged_entities = merge_by_normalized_name(entities)
    return merged_entities
```

**Entity Types** (from spaCy):

- `PERSON`: People (e.g., "John Stuart Mill", "Kant")
- `ORG`: Organizations (e.g., "Equifax", "Nike", "FDA")
- `GPE`: Geopolitical entities (e.g., "United States", "Thailand")
- `DATE`: Dates (e.g., "2017", "June 23")
- `MONEY`: Monetary values (e.g., "$1M", "5 billion dollars")
- `PERCENT`: Percentages (e.g., "25%", "half")
- `PRODUCT`: Products (e.g., "iPhone", "Windows")
- `EVENT`: Named events (e.g., "World War II", "Equifax Data Breach")
- `LAW`: Laws/regulations (e.g., "GDPR", "Sarbanes-Oxley")
- `CONCEPT`: Fallback for regex-extracted concepts

**Cross-Document Entity Resolution**:

```python
# Example: "Equifax" mentioned in 3 different documents
# All chunks link to the SAME entity node via normalized_name
Entity(entity_name="Equifax", normalized_name="equifax", entity_type="ORG")
```

---

### **Step 5: Topic Extraction**

**File**: `backend/app/services/ingestion/pipeline.py` → `_extract_topics_from_chunks()`

**Purpose**: Create high-level thematic nodes for cross-document navigation.

**Strategy**:

```python
def extract_topics(chunks, sections, entities):
    topics = []

    # 1. Topics from section titles
    for section in sections:
        title = section['title'].lower()
        # Extract noun phrases as topics
        keywords = extract_keywords(title)
        if keywords:
            topics.append({
                "topic_name": title,
                "normalized_name": normalize(title),
                "keywords": keywords,
            })

    # 2. Topics from entity clusters
    entity_types = {}
    for entity in entities:
        etype = entity['entity_type']
        if etype not in entity_types:
            entity_types[etype] = []
        entity_types[etype].append(entity['entity_name'])

    # Map entity types to topics
    topic_mapping = {
        "ORG": "organizations and companies",
        "PERSON": "people and individuals",
        "GPE": "geographic locations",
        "DATE": "timeline and events",
        "MONEY": "financial information",
        "PERCENT": "statistics and metrics",
    }

    for etype, topic_name in topic_mapping.items():
        if etype in entity_types:
            topics.append({
                "topic_name": topic_name,
                "normalized_name": topic_name.lower(),
                "keywords": list(set([e.lower() for e in entity_types[etype]][:20])),
            })

    return deduplicate_topics(topics)
```

**Example Topics**:

```
Topic: "business ethics"
  - keywords: ["ethics", "business", "moral", "stakeholder"]
  - linked_entities: ["utilitarianism", "deontology"]
  - linked_chunks: [chunk_1, chunk_2, ..., chunk_50]

Topic: "organizations and companies"
  - keywords: ["equifax", "nike", "facebook", "kroger"]
  - linked_entities: [Entity("Equifax"), Entity("Nike")]
  - linked_chunks: [chunk_10, chunk_25, ...]
```

---

### **Step 6: Entity Relationship Extraction**

**File**: `backend/app/services/ingestion/pipeline.py` → `_extract_entity_relationships()`

**Purpose**: Create `Entity --RELATED_TO--> Entity` relationships based on co-occurrence.

**Algorithm**:

```python
def extract_entity_relationships(entities, chunks):
    relationships = []

    # Build entity-to-chunk index
    entity_chunks = defaultdict(set)
    for entity in entities:
        for chunk_id in entity['chunk_ids']:
            entity_chunks[entity['normalized_name']].add(chunk_id)

    # Find entity pairs that co-occur in the same chunk
    for entity1 in entities:
        for entity2 in entities:
            if entity1['normalized_name'] >= entity2['normalized_name']:
                continue  # Avoid duplicates

            # Count co-occurrences
            shared_chunks = entity_chunks[entity1['normalized_name']] & \
                           entity_chunks[entity2['normalized_name']]

            if len(shared_chunks) >= 2:  # Threshold: appear together in 2+ chunks
                relationships.append({
                    "from_entity": entity1['normalized_name'],
                    "to_entity": entity2['normalized_name'],
                    "relationship_type": "RELATED_TO",
                    "frequency": len(shared_chunks),
                })

    return relationships
```

**Example**:

```cypher
// "Equifax" and "data breach" co-occur in 15 chunks
(:Entity {normalized_name: "equifax"})
  -[:RELATED_TO {frequency: 15}]->
(:Entity {normalized_name: "data breach"})

// Enables multi-hop queries:
// "Equifax" -> "data breach" -> "privacy" -> "ethical responsibility"
```

---

## **Graph Schema & Data Model**

### **Node Types**

```cypher
// 1. Document (root node)
CREATE (:Document {
    document_id: "uuid",          // Unique identifier
    title: "Business Ethics.pdf",
    source: "uploads/...",
    document_type: "pdf",
    created_at: "2025-11-30T..."
})

// 2. Section (logical divisions)
CREATE (:Section {
    section_id: "uuid",
    section_title: "2.4 Utilitarianism: The Greatest Good",
    section_index: 5
})

// 3. Chunk (retrieval unit)
CREATE (:Chunk {
    chunk_id: "uuid",
    chunk_index: 42,
    chunk_type: "text",  // or "table", "image"
    content: "Utilitarianism is a consequentialist ethical theory..."
})

// 4. Entity (named entities from spaCy)
CREATE (:Entity {
    entity_id: "uuid",
    entity_name: "Equifax",
    normalized_name: "equifax",    // For cross-doc matching
    entity_type: "ORG",
    confidence: 1.0
})

// 5. Topic (thematic ontology)
CREATE (:Topic {
    topic_id: "uuid",
    topic_name: "Business Ethics",
    normalized_name: "business ethics",
    keywords: ["ethics", "moral", "business", "stakeholder"]
})

// 6. Media (images, tables, charts)
CREATE (:Media {
    media_id: "uuid",
    media_type: "IMAGE",
    media_url: "s3://...",
    caption: "Figure 1: Ethical Framework"
})
```

### **Relationship Types**

```cypher
// Hierarchical structure
(:Document)-[:HAS_SECTION {order: 1}]->(:Section)
(:Section)-[:HAS_CHUNK {order: 1, chunk_index: 0}]->(:Chunk)

// Sequential links (for context expansion)
(:Chunk)-[:NEXT_CHUNK]->(:Chunk)

// Entity connections
(:Chunk)-[:MENTIONS {frequency: 5, importance: 0.8}]->(:Entity)
(:Section)-[:ABOUT {importance: 0.9}]->(:Entity)

// Entity relationships (co-occurrence)
(:Entity)-[:RELATED_TO {frequency: 10}]->(:Entity)

// Topic connections (cross-document navigation)
(:Chunk)-[:HAS_TOPIC {relevance: 0.8}]->(:Topic)
(:Entity)-[:ASSOCIATED_WITH {relevance: 0.7}]->(:Topic)

// Media connections
(:Chunk)-[:HAS_MEDIA {position: "inline", relevance: 0.9}]->(:Media)
(:Media)-[:DESCRIBES {relevance: 0.8}]->(:Entity)
```

---

## **Graph Construction**

**File**: `backend/app/repositories/graph_repository.py`

### **Batch Creation Process**

```python
def create_document_graph_batch(document_id, title, source, sections):
    """
    Creates the complete graph structure for a document in batches.

    Process:
    1. Create Document node
    2. For each section:
       a. Create Section node
       b. Link Document --HAS_SECTION--> Section
       c. For each chunk in section:
          - Create Chunk node
          - Link Section --HAS_CHUNK--> Chunk
    3. Create NEXT_CHUNK relationships (sequential)
    4. Create Entity nodes (with normalized_name for merging)
    5. Create Chunk --MENTIONS--> Entity relationships
    6. Create Entity --RELATED_TO--> Entity relationships
    7. Create Topic nodes
    8. Create Chunk --HAS_TOPIC--> Topic relationships
    9. Create Entity --ASSOCIATED_WITH--> Topic relationships
    """

    # Step 1: Document node
    driver = get_neo4j_driver()
    session = driver.session(database="neo4j")

    session.run("""
        MERGE (d:Document {document_id: $doc_id})
        SET d.title = $title,
            d.source = $source
    """, doc_id=document_id, title=title, source=source)

    # Steps 2-3: Sections and chunks
    for section in sections:
        session.run("""
            MATCH (d:Document {document_id: $doc_id})
            MERGE (s:Section {section_id: $sec_id})
            SET s.section_title = $title,
                s.section_index = $index
            MERGE (d)-[:HAS_SECTION {order: $index}]->(s)
        """, doc_id=document_id, sec_id=section['section_id'],
             title=section['title'], index=section['index'])

        for chunk_data in section['chunks']:
            session.run("""
                MATCH (s:Section {section_id: $sec_id})
                MERGE (c:Chunk {chunk_id: $chunk_id})
                SET c.content = $content,
                    c.chunk_index = $chunk_index,
                    c.chunk_type = $chunk_type
                MERGE (s)-[:HAS_CHUNK {order: $chunk_index}]->(c)
            """, sec_id=section['section_id'],
                 chunk_id=chunk_data['chunk_id'],
                 content=chunk_data['content'],
                 chunk_index=chunk_data['chunk_index'],
                 chunk_type=chunk_data['chunk_type'])

    session.close()
```

### **Entity Node Creation (Cross-Document Merging)**

```python
def create_entity_nodes_batch(entities):
    """
    Create entity nodes with automatic cross-document merging.

    Key: Uses MERGE on normalized_name, so entities with the same
    normalized name share a single node across all documents.
    """
    session = driver.session(database="neo4j")

    session.run("""
        UNWIND $entities AS entity
        MERGE (e:Entity {normalized_name: entity.normalized_name})
        ON CREATE SET
            e.entity_id = entity.entity_id,
            e.entity_name = entity.entity_name,
            e.entity_type = entity.entity_type,
            e.confidence = entity.confidence
        ON MATCH SET
            e.entity_name = CASE
                WHEN length(entity.entity_name) > length(e.entity_name)
                THEN entity.entity_name
                ELSE e.entity_name
            END
    """, entities=entities)
```

**Result**: All mentions of "Equifax" across 10 documents link to the **same** Entity node.

---

## **Query Processing & Retrieval**

### **Graph Retrieval Strategy (Multi-Tiered)**

**File**: `backend/app/services/retrieval/graph_retriever.py` → `query_by_keywords()`

**Philosophy**: Use **multiple complementary strategies** to find relevant chunks.

#### **Strategy 1: Section-Title Search** (Primary)

**Priority**: 🔥 **Highest** (score: 0.95)

**Purpose**: Find sections whose titles match query keywords, then return **ALL chunks** from those sections.

**Why**: When a user asks "Explain utilitarianism", they want the _explanatory content_ from sections titled "Utilitarianism", not just passing mentions.

**Cypher Query**:

```cypher
MATCH (d:Document)
      -[:HAS_SECTION]->(s:Section)
      -[:HAS_CHUNK]->(c:Chunk)
WHERE ANY(keyword IN $keywords
          WHERE toLower(s.section_title) CONTAINS toLower(keyword))
WITH DISTINCT s, d, c
ORDER BY s.section_index, c.chunk_index
RETURN c.chunk_id, c.content, c.chunk_index, c.chunk_type,
       d.document_id, d.title AS filename,
       s.section_title, s.section_index
LIMIT $limit
```

**Example**:

```
Query: "Summarize utilitarianism, deontology, and virtue ethics"
Keywords: ["utilitarianism", "deontology", "virtue", "ethics"]

Matches:
- Section: "2.4 Utilitarianism: The Greatest Good" → 20 chunks
- Section: "2.5 Deontology: Ethics as Duty" → 15 chunks
- Section: "2.3 Comparing Virtue Ethics..." → 18 chunks

Total: 53 chunks (top 20 returned)
```

---

#### **Strategy 2: Entity Mention Search** (Secondary)

**Priority**: 🟠 **Medium** (score: 0.7)

**Purpose**: Find chunks that mention entities matching query keywords.

**Cypher Query**:

```cypher
MATCH (e:Entity)
      <-[:MENTIONS]-(c:Chunk)
      <-[:HAS_CHUNK]-(s:Section)
      <-[:HAS_SECTION]-(d:Document)
WHERE ANY(keyword IN $keywords
          WHERE toLower(e.entity_name) CONTAINS toLower(keyword))
WITH DISTINCT c, d, s, count(DISTINCT e) AS match_count
RETURN c.chunk_id, c.content, c.chunk_index, c.chunk_type,
       d.document_id, d.title AS filename,
       s.section_title, s.section_index, match_count
ORDER BY match_count DESC, s.section_index, c.chunk_index
LIMIT $limit
```

**Example**:

```
Query: "Equifax data breach ethical failures"
Keywords: ["equifax", "data", "breach", "ethical", "failures"]

Entity Matches:
- Entity("Equifax", type=ORG) → 12 chunks
- Entity("data breach", type=EVENT) → 8 chunks
```

---

#### **Strategy 3: Topic-Based Search** (Cross-Document)

**Priority**: 🟡 **Medium** (score: 0.75)

**Purpose**: Navigate across documents via shared topics.

**Cypher Query**:

```cypher
MATCH (t:Topic)
      <-[:HAS_TOPIC]-(c:Chunk)
      <-[:HAS_CHUNK]-(s:Section)
      <-[:HAS_SECTION]-(d:Document)
WHERE ANY(keyword IN $keywords
          WHERE toLower(t.topic_name) CONTAINS toLower(keyword)
             OR ANY(kw IN t.keywords WHERE toLower(kw) CONTAINS toLower(keyword)))
WITH DISTINCT c, d, s, t
RETURN c.chunk_id, c.content, c.chunk_index, c.chunk_type,
       d.document_id, d.title AS filename,
       s.section_title, s.section_index, t.topic_name
LIMIT $limit
```

**Use Case**: "corporate social responsibility" across 5 different documents.

---

#### **Strategy 4: Entity Relationship Traversal** (Multi-Hop)

**Priority**: 🔵 **Low** (score: 0.6)

**Purpose**: Find indirectly related content through entity connections.

**Cypher Query**:

```cypher
MATCH path = (e1:Entity)-[:RELATED_TO*1..2]-(e2:Entity)
WHERE toLower(e1.entity_name) IN $entity_names
WITH DISTINCT e2
MATCH (c:Chunk)-[:MENTIONS]->(e2)
      <-[:HAS_CHUNK]-(s:Section)
      <-[:HAS_SECTION]-(d:Document)
RETURN c.chunk_id, c.content, c.chunk_index, c.chunk_type,
       d.document_id, d.title AS filename,
       s.section_title, s.section_index
LIMIT $limit
```

**Example**:

```
Query: "Equifax"
Direct: Equifax entity → 12 chunks
1-hop: Equifax -[:RELATED_TO]-> "data breach" → 8 chunks
2-hop: "data breach" -[:RELATED_TO]-> "privacy" → 15 chunks
```

---

#### **Strategy 5: Content Search** (Fallback - ALWAYS RUNS)

**Priority**: 🟢 **Always Run** (score: 0.75)

**Purpose**: Direct keyword matching in chunk content (catches everything else).

**Cypher Query**:

```cypher
MATCH (d:Document)
      -[:HAS_SECTION]->(s:Section)
      -[:HAS_CHUNK]->(c:Chunk)
WITH c, d, s,
     size([keyword IN $keywords
           WHERE toLower(c.content) CONTAINS toLower(keyword)]) AS match_count
WHERE match_count > 0
RETURN c.chunk_id, c.content, c.chunk_index, c.chunk_type,
       d.document_id, d.title AS filename,
       s.section_title, s.section_index, match_count
ORDER BY match_count DESC, s.section_index, c.chunk_index
LIMIT $limit
```

**Why Always Run**: Ensures we find content even if:

- Entity extraction failed
- Section detection was imperfect
- Keywords don't match entity/topic names exactly

---

### **Context Expansion (NEXT_CHUNK Traversal)**

**Purpose**: Expand retrieved chunks with surrounding context.

**Implementation**:

```cypher
// Get initial chunk + 2 preceding + 2 following chunks
MATCH (initial:Chunk {chunk_id: $chunk_id})
OPTIONAL MATCH (before2:Chunk)-[:NEXT_CHUNK]->(before1:Chunk)-[:NEXT_CHUNK]->(initial)
OPTIONAL MATCH (initial)-[:NEXT_CHUNK]->(after1:Chunk)-[:NEXT_CHUNK]->(after2:Chunk)
RETURN before2, before1, initial, after1, after2
```

**Example**:

```
Initial retrieval: Chunk #42 (mentions "utilitarianism")
Context expansion:
  Chunk #40: "...ethical theories can be categorized..."
  Chunk #41: "...the first major category is consequentialism..."
  Chunk #42: "Utilitarianism is a consequentialist theory that..."  ← Initial hit
  Chunk #43: "...Jeremy Bentham and John Stuart Mill developed..."
  Chunk #44: "...the greatest good for the greatest number..."

Result: User gets full explanation, not just the mention.
```

---

## **Hybrid Retrieval Integration**

**File**: `backend/app/services/retrieval/hybrid_retriever.py`

### **Retrieval Flow**

```python
async def retrieve(query, limit=10,
                  enable_sparse=True,
                  enable_dense=True,
                  enable_graph=True):
    """
    Parallel hybrid retrieval combining multiple methods.

    Process:
    1. Generate query embedding (for dense/vector search)
    2. Run retrievers in parallel:
       - BM25 sparse retrieval (Elasticsearch)
       - Dense vector retrieval (Qdrant text chunks)
       - Table vector retrieval (Qdrant table chunks)
       - Image vector retrieval (Qdrant image chunks)
       - Graph retrieval (Neo4j)
    3. Normalize scores (min-max normalization)
    4. Merge results with weighted scoring
    5. Deduplicate by chunk_id
    6. Return top-K
    """

    # Step 1: Generate query embedding (1536-dim for text)
    query_embedding = await text_embedder.embed(query)

    # Step 2: Parallel retrieval
    results = await asyncio.gather(
        retrieve_sparse(query),        # BM25
        retrieve_dense(query_embedding), # Vector
        retrieve_graph(query),          # Graph
    )

    sparse_chunks, dense_chunks, graph_chunks = results

    # Step 3: Normalize scores to [0, 1]
    sparse_chunks = normalize_scores(sparse_chunks)
    dense_chunks = normalize_scores(dense_chunks)
    graph_chunks = normalize_scores(graph_chunks)

    # Step 4: Merge with weights
    weights = {
        "sparse": 0.3,   # BM25 weight
        "dense": 0.4,    # Vector weight
        "graph": 0.3,    # Graph weight
    }

    merged_chunks = merge_results(
        sparse_chunks, dense_chunks, graph_chunks,
        weights=weights
    )

    # Step 5: Return top-K
    return merged_chunks[:limit]
```

### **Score Normalization**

```python
def normalize_scores(chunks):
    """
    Min-max normalization: scale scores to [0, 1] range.

    This ensures fair comparison across different retrieval methods
    (BM25 scores are typically 1-20, vector scores are 0.5-0.99, etc.)
    """
    if not chunks:
        return chunks

    scores = [c['score'] for c in chunks]
    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        # All scores are equal
        for c in chunks:
            c['normalized_score'] = 1.0
    else:
        for c in chunks:
            c['normalized_score'] = (c['score'] - min_score) / (max_score - min_score)

    return chunks
```

### **Weighted Merging**

```python
def merge_results(sparse_chunks, dense_chunks, graph_chunks, weights):
    """
    Merge results from multiple retrievers using weighted averaging.

    For chunks that appear in multiple retrievers, combine scores:
    final_score = w1 * sparse_score + w2 * dense_score + w3 * graph_score
    """
    chunk_scores = {}  # chunk_id -> {scores, metadata}

    for chunk in sparse_chunks:
        cid = chunk['chunk_id']
        if cid not in chunk_scores:
            chunk_scores[cid] = {'chunk': chunk, 'scores': {}}
        chunk_scores[cid]['scores']['sparse'] = chunk['normalized_score']

    for chunk in dense_chunks:
        cid = chunk['chunk_id']
        if cid not in chunk_scores:
            chunk_scores[cid] = {'chunk': chunk, 'scores': {}}
        chunk_scores[cid]['scores']['dense'] = chunk['normalized_score']

    for chunk in graph_chunks:
        cid = chunk['chunk_id']
        if cid not in chunk_scores:
            chunk_scores[cid] = {'chunk': chunk, 'scores': {}}
        chunk_scores[cid]['scores']['graph'] = chunk['normalized_score']

    # Calculate final weighted scores
    merged = []
    for cid, data in chunk_scores.items():
        final_score = (
            data['scores'].get('sparse', 0) * weights['sparse'] +
            data['scores'].get('dense', 0) * weights['dense'] +
            data['scores'].get('graph', 0) * weights['graph']
        )
        chunk = data['chunk']
        chunk['final_score'] = final_score
        merged.append(chunk)

    # Sort by final score
    merged.sort(key=lambda x: x['final_score'], reverse=True)
    return merged
```

---

## **Retrieval Scenarios**

### **Scenario 1: Graph-Only Retrieval**

**Configuration**: `enable_graph=True`, all others `False`

**Query**: "Summarize utilitarianism, deontology, and virtue ethics"

**Process**:

```
1. Extract keywords: ["utilitarianism", "deontology", "virtue", "ethics"]
2. Section-title search:
   - "2.4 Utilitarianism: The Greatest Good" → 20 chunks (score: 0.95)
   - "2.5 Deontology: Ethics as Duty" → 15 chunks (score: 0.95)
3. Content search (fallback):
   - 9 additional chunks mentioning keywords (score: 0.75)
4. Sort by score, return top 10
```

**Result**: 10 chunks from relevant sections (rich, explanatory content)

**Strengths**:

- ✅ Gets complete sections (not just snippets)
- ✅ Preserves document structure
- ✅ Fast (single query to Neo4j)

**Weaknesses**:

- ⚠️ Depends on good section detection
- ⚠️ May miss content if keywords don't match section titles or entities

---

### **Scenario 2: Hybrid Retrieval (All Enabled)**

**Configuration**: `enable_sparse=True`, `enable_dense=True`, `enable_graph=True`

**Query**: "What were the ethical failures in the Equifax data breach?"

**Process**:

```
1. BM25 sparse retrieval:
   - Matches: "ethical", "failures", "equifax", "data", "breach"
   - Top 20 chunks (BM25 scores: 5.2 to 12.8)

2. Vector dense retrieval:
   - Semantic similarity to query embedding
   - Top 20 chunks (cosine similarity: 0.72 to 0.89)

3. Graph retrieval:
   - Section-title: "9. Describe how a company's ethical..." → 20 chunks
   - Entity: Entity("Equifax") → 12 chunks
   - Content: 15 chunks mentioning keywords

4. Normalize scores to [0, 1]:
   - BM25: [0.0, 0.41, 0.53, ..., 1.0]
   - Vector: [0.0, 0.28, 0.35, ..., 1.0]
   - Graph: [0.60, 0.75, 0.95, ..., 1.0]

5. Merge with weights (sparse: 0.3, dense: 0.4, graph: 0.3):
   - Chunk A: 0.3*0.8 + 0.4*0.9 + 0.3*0.95 = 0.885
   - Chunk B: 0.3*0.5 + 0.4*0.7 + 0.3*0.75 = 0.655

6. Return top 10 by final score
```

**Result**: Diverse chunks combining:

- Keyword-matched content (BM25)
- Semantically similar content (vector)
- Structurally relevant content (graph)

**Strengths**:

- ✅ Most comprehensive retrieval
- ✅ Combines complementary methods
- ✅ Handles varied query types

**Weaknesses**:

- ⚠️ More expensive (3 database queries)
- ⚠️ Requires careful weight tuning

---

## **Performance & Optimization**

### **Indexing Strategy**

**Neo4j Indexes** (for fast lookups):

```cypher
// Primary keys
CREATE INDEX ON :Document(document_id)
CREATE INDEX ON :Section(section_id)
CREATE INDEX ON :Chunk(chunk_id)
CREATE INDEX ON :Entity(entity_id)
CREATE INDEX ON :Entity(normalized_name)  // Cross-doc resolution
CREATE INDEX ON :Topic(topic_id)
CREATE INDEX ON :Topic(normalized_name)

// Search indexes
CREATE INDEX ON :Section(section_title)
CREATE INDEX ON :Entity(entity_name)
CREATE INDEX ON :Topic(topic_name)

// Full-text search (for content)
CALL db.index.fulltext.createNodeIndex(
    "chunk_content",
    ["Chunk"],
    ["content"]
)
```

### **Batch Processing**

**Large Document Handling** (500+ chunks):

```python
# Process in batches to avoid memory issues
BATCH_SIZE = 500

if total_chunks > BATCH_SIZE:
    for i in range(0, total_chunks, BATCH_SIZE):
        batch_sections = sections[i:i+BATCH_SIZE]
        create_document_graph_batch(document_id, title, source, batch_sections)
```

### **Query Optimization**

**Parallel Retrieval**:

- BM25, Vector, and Graph queries run **simultaneously** using `asyncio.gather()`
- Saves ~60% time vs sequential retrieval

**Caching**:

- Query embeddings cached (same query → reuse embedding)
- Entity normalized names cached (for fast merging)

### **Metrics**

**Key Metrics Tracked**:

- `neo4j_query_duration_seconds`: Query latency
- `neo4j_chunks_retrieved_via_graph`: Throughput
- `retrieval_duration_seconds{retrieval_type="graph"}`: Graph-specific latency
- `hybrid_merge_duration_seconds`: Merge overhead

---

## **Current Status & Future Improvements**

### **✅ Implemented**

1. ✅ Multi-tiered graph retrieval (section, entity, topic, content)
2. ✅ spaCy NER for universal entity extraction
3. ✅ Cross-document entity resolution (via `normalized_name`)
4. ✅ Topic-based cross-document navigation
5. ✅ Entity relationship extraction (`RELATED_TO`)
6. ✅ Sequential context expansion (`NEXT_CHUNK`)
7. ✅ Hybrid retrieval with score normalization

### **⚠️ Known Issues**

1. ⚠️ **Section detection too aggressive**: Treats answers/footnotes as headings

   - **Fix**: Add filters for lines >100 chars, ending with periods, etc.

2. ⚠️ **Graph retrieval sometimes finds wrong sections**:

   - Example: "Equifax" query matches "9. Describe how a company's **e**thical..." (substring match)
   - **Fix**: Use more precise keyword matching (word boundaries)

3. ⚠️ **Topic relationships not always created**:
   - Bug fixed: `entities` variable undefined if entity extraction failed
   - **Status**: Fixed in latest version

### **🔮 Future Enhancements**

1. **Graph Embeddings**: Add vector embeddings to Entity/Topic nodes for semantic search within graph
2. **Temporal Relationships**: Add `PRECEDED_BY`, `FOLLOWED_BY` for timeline queries
3. **Document Similarity**: Link similar documents via shared entity/topic patterns
4. **Query Decomposition**: Break complex queries into sub-queries, combine results
5. **Feedback Loop**: Use retrieval success/failure to refine entity extraction and section detection
6. **Multi-lingual Support**: spaCy supports 20+ languages (currently only English)

---

## **Summary**

This GraphRAG system provides **structure-aware, context-rich retrieval** by:

1. **Ingestion**: Extract text → Detect sections → Chunk → Extract entities/topics → Build graph
2. **Storage**: Multi-index strategy (Neo4j graph + Qdrant vector + Elasticsearch BM25)
3. **Retrieval**: Multi-tiered graph search (section → entity → topic → content) with context expansion
4. **Hybrid**: Combine graph + vector + sparse retrieval with weighted scoring

**Key Innovation**: Uses **document structure** (sections) and **entity relationships** (co-occurrence) to provide answers that are **contextually complete**, not just keyword-matched snippets.

**Universal Design**: Works across any domain via spaCy NER (no domain-specific ontologies required).

---

**Save this as**: `graph_rag_architecture.md`
