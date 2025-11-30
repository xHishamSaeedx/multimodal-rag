"""
Graph retriever for querying Neo4j knowledge graph.

This module provides methods for retrieving chunks from the knowledge graph
based on entities, sections, and document structure.

Graph Retrieval Strategy (following GraphRAG best practices):
1. Section-aware retrieval: Find relevant sections by title/keyword, then get ALL chunks from those sections
2. Entity-based retrieval: Find chunks mentioning specific entities
3. Multi-hop traversal: Navigate from initial hits to related chunks through entity connections
4. Cross-document discovery: Find related content across documents through shared entities

Key insight: When querying for a concept like "Utilitarianism", we need to:
- Find sections with matching titles (e.g., "2.4 Utilitarianism: The Greatest Good")
- Return ALL chunks from those sections (the actual explanatory content)
- Also find chunks that mention the keyword in their content
"""
import time
import re
from typing import List, Dict, Any, Optional
from neo4j import Driver

from app.core.neo4j_database import get_neo4j_driver
from app.core.config import settings
from app.core.database import DatabaseError
from app.repositories.graph_schema import NODE_LABELS, PROPERTY_KEYS, RELATIONSHIP_TYPES
from app.utils.metrics import (
    neo4j_graph_queries_total,
    neo4j_graph_query_duration_seconds,
    neo4j_chunks_retrieved_via_graph,
    neo4j_query_duration_seconds,
    neo4j_queries_total,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)


class GraphRetriever:
    """
    Retrieve chunks from Neo4j knowledge graph.
    """
    
    def __init__(self, driver: Optional[Driver] = None):
        """
        Initialize graph retriever.
        
        Args:
            driver: Optional Neo4j driver instance (creates new if not provided)
        """
        self.driver = driver
        self.database = settings.neo4j_database
    
    def _get_driver(self) -> Driver:
        """Get Neo4j driver instance."""
        if self.driver is not None:
            return self.driver
        return get_neo4j_driver()
    
    def diagnose_graph_content(self, document_id: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
        """
        Diagnostic method to check if content is being stored correctly in the graph.
        
        Returns statistics about chunks with/without content for debugging.
        
        Args:
            document_id: Optional - limit to specific document
            limit: Number of sample chunks to inspect
        
        Returns:
            Dictionary with diagnostic info
        """
        driver = self._get_driver()
        session = driver.session(database=self.database)
        
        try:
            # Count total chunks
            if document_id:
                count_query = f"""
                MATCH (d:{NODE_LABELS['DOCUMENT']} {{{PROPERTY_KEYS['DOCUMENT_ID']}: $doc_id}})
                      -[:{RELATIONSHIP_TYPES['HAS_SECTION']}]->(s:{NODE_LABELS['SECTION']})
                      -[:{RELATIONSHIP_TYPES['HAS_CHUNK']}]->(c:{NODE_LABELS['CHUNK']})
                RETURN count(c) AS total_chunks,
                       sum(CASE WHEN c.{PROPERTY_KEYS['CONTENT']} IS NOT NULL AND c.{PROPERTY_KEYS['CONTENT']} <> '' THEN 1 ELSE 0 END) AS with_content,
                       sum(CASE WHEN c.{PROPERTY_KEYS['CONTENT']} IS NULL OR c.{PROPERTY_KEYS['CONTENT']} = '' THEN 1 ELSE 0 END) AS without_content
                """
                result = session.run(count_query, doc_id=document_id)
            else:
                count_query = f"""
                MATCH (c:{NODE_LABELS['CHUNK']})
                RETURN count(c) AS total_chunks,
                       sum(CASE WHEN c.{PROPERTY_KEYS['CONTENT']} IS NOT NULL AND c.{PROPERTY_KEYS['CONTENT']} <> '' THEN 1 ELSE 0 END) AS with_content,
                       sum(CASE WHEN c.{PROPERTY_KEYS['CONTENT']} IS NULL OR c.{PROPERTY_KEYS['CONTENT']} = '' THEN 1 ELSE 0 END) AS without_content
                """
                result = session.run(count_query)
            
            counts = result.single()
            
            # Get sample sections
            if document_id:
                section_query = f"""
                MATCH (d:{NODE_LABELS['DOCUMENT']} {{{PROPERTY_KEYS['DOCUMENT_ID']}: $doc_id}})
                      -[:{RELATIONSHIP_TYPES['HAS_SECTION']}]->(s:{NODE_LABELS['SECTION']})
                RETURN s.{PROPERTY_KEYS['SECTION_TITLE']} AS title,
                       s.{PROPERTY_KEYS['SECTION_INDEX']} AS idx
                ORDER BY s.{PROPERTY_KEYS['SECTION_INDEX']}
                LIMIT 20
                """
                section_result = session.run(section_query, doc_id=document_id)
            else:
                section_query = f"""
                MATCH (s:{NODE_LABELS['SECTION']})
                RETURN DISTINCT s.{PROPERTY_KEYS['SECTION_TITLE']} AS title,
                       s.{PROPERTY_KEYS['SECTION_INDEX']} AS idx
                ORDER BY s.{PROPERTY_KEYS['SECTION_INDEX']}
                LIMIT 20
                """
                section_result = session.run(section_query)
            
            sections = [{"title": r["title"], "index": r["idx"]} for r in section_result]
            
            # Get sample chunks with content info
            if document_id:
                sample_query = f"""
                MATCH (d:{NODE_LABELS['DOCUMENT']} {{{PROPERTY_KEYS['DOCUMENT_ID']}: $doc_id}})
                      -[:{RELATIONSHIP_TYPES['HAS_SECTION']}]->(s:{NODE_LABELS['SECTION']})
                      -[:{RELATIONSHIP_TYPES['HAS_CHUNK']}]->(c:{NODE_LABELS['CHUNK']})
                RETURN c.{PROPERTY_KEYS['CHUNK_ID']} AS chunk_id,
                       s.{PROPERTY_KEYS['SECTION_TITLE']} AS section,
                       CASE WHEN c.{PROPERTY_KEYS['CONTENT']} IS NOT NULL THEN size(c.{PROPERTY_KEYS['CONTENT']}) ELSE 0 END AS content_length,
                       left(c.{PROPERTY_KEYS['CONTENT']}, 100) AS content_preview
                ORDER BY s.{PROPERTY_KEYS['SECTION_INDEX']}, c.{PROPERTY_KEYS['CHUNK_INDEX']}
                LIMIT $limit
                """
                sample_result = session.run(sample_query, doc_id=document_id, limit=limit)
            else:
                sample_query = f"""
                MATCH (s:{NODE_LABELS['SECTION']})-[:{RELATIONSHIP_TYPES['HAS_CHUNK']}]->(c:{NODE_LABELS['CHUNK']})
                RETURN c.{PROPERTY_KEYS['CHUNK_ID']} AS chunk_id,
                       s.{PROPERTY_KEYS['SECTION_TITLE']} AS section,
                       CASE WHEN c.{PROPERTY_KEYS['CONTENT']} IS NOT NULL THEN size(c.{PROPERTY_KEYS['CONTENT']}) ELSE 0 END AS content_length,
                       left(c.{PROPERTY_KEYS['CONTENT']}, 100) AS content_preview
                LIMIT $limit
                """
                sample_result = session.run(sample_query, limit=limit)
            
            samples = []
            for r in sample_result:
                samples.append({
                    "chunk_id": r["chunk_id"][:8] + "..." if r["chunk_id"] else None,
                    "section": r["section"],
                    "content_length": r["content_length"],
                    "content_preview": r["content_preview"][:50] + "..." if r["content_preview"] and len(r["content_preview"]) > 50 else r["content_preview"],
                })
            
            return {
                "total_chunks": counts["total_chunks"] if counts else 0,
                "chunks_with_content": counts["with_content"] if counts else 0,
                "chunks_without_content": counts["without_content"] if counts else 0,
                "sections": sections,
                "sample_chunks": samples,
                "document_id": document_id,
            }
        except Exception as e:
            logger.error(f"Diagnostic query failed: {str(e)}")
            return {"error": str(e)}
        finally:
            session.close()
    
    def query_by_entity(
        self,
        entity_name: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Query chunks that mention a specific entity.
        
        Args:
            entity_name: Name of the entity to search for
            limit: Maximum number of chunks to return
        
        Returns:
            List of chunk dictionaries with chunk_id, document_id, etc.
        """
        query = f"""
        MATCH (e:{NODE_LABELS['ENTITY']} {{{PROPERTY_KEYS['ENTITY_NAME']}: $entity_name}})
              <-[:{RELATIONSHIP_TYPES['MENTIONS']}]-(c:{NODE_LABELS['CHUNK']})
              <-[:{RELATIONSHIP_TYPES['HAS_CHUNK']}]-(s:{NODE_LABELS['SECTION']})
              <-[:{RELATIONSHIP_TYPES['HAS_SECTION']}]-(d:{NODE_LABELS['DOCUMENT']})
        RETURN c.{PROPERTY_KEYS['CHUNK_ID']} AS chunk_id,
               c.{PROPERTY_KEYS['CHUNK_INDEX']} AS chunk_index,
               c.{PROPERTY_KEYS['CHUNK_TYPE']} AS chunk_type,
               c.{PROPERTY_KEYS['CONTENT']} AS chunk_text,
               d.{PROPERTY_KEYS['DOCUMENT_ID']} AS document_id,
               d.{PROPERTY_KEYS['TITLE']} AS filename,
               s.{PROPERTY_KEYS['SECTION_TITLE']} AS section_title,
               s.{PROPERTY_KEYS['SECTION_INDEX']} AS section_index
        ORDER BY s.{PROPERTY_KEYS['SECTION_INDEX']}, c.{PROPERTY_KEYS['CHUNK_INDEX']}
        LIMIT $limit
        """
        
        query_start = time.time()
        driver = self._get_driver()
        session = driver.session(database=self.database)
        try:
            result = session.run(query, entity_name=entity_name, limit=limit)
            chunks = []
            for record in result:
                chunks.append({
                    "chunk_id": record["chunk_id"],
                    "document_id": record["document_id"],
                    "chunk_index": record["chunk_index"],
                    "chunk_type": record.get("chunk_type", "text"),
                    "chunk_text": record.get("chunk_text", ""),  # Content from Neo4j
                    "filename": record.get("filename", "Unknown Document"),
                    "metadata": {
                        "section_title": record.get("section_title"),
                        "section_index": record.get("section_index"),
                    }
                })
            
            # Record metrics
            query_duration = time.time() - query_start
            neo4j_graph_query_duration_seconds.labels(query_type="by_entity").observe(query_duration)
            neo4j_graph_queries_total.labels(query_type="by_entity").inc()
            neo4j_chunks_retrieved_via_graph.labels(query_type="by_entity").inc(len(chunks))
            neo4j_query_duration_seconds.labels(operation="read").observe(query_duration)
            neo4j_queries_total.labels(operation="read", status="success").inc()
            
            return chunks
        except Exception as e:
            query_duration = time.time() - query_start
            neo4j_graph_queries_total.labels(query_type="by_entity").inc()
            neo4j_query_duration_seconds.labels(operation="read").observe(query_duration)
            neo4j_queries_total.labels(operation="read", status="error").inc()
            logger.error(f"Graph query by entity failed: {str(e)}")
            raise
        finally:
            session.close()
    
    def query_by_section_title(
        self,
        keywords: List[str],
        limit: int = 20,
        document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query ALL chunks from sections whose titles match keywords.
        
        This is the KEY method for retrieving actual content about a topic.
        When searching for "Utilitarianism", this finds sections titled
        "2.4 Utilitarianism: The Greatest Good" and returns ALL chunks
        in that section - the actual explanatory content.
        
        Args:
            keywords: List of keywords to search for in section titles
            limit: Maximum number of chunks to return
            document_id: Optional document ID to filter results
        
        Returns:
            List of chunk dictionaries with full content
        """
        query_start = time.time()
        driver = self._get_driver()
        session = driver.session(database=self.database)
        try:
            # Build keyword match pattern for section titles
            if document_id:
                query = f"""
                MATCH (d:{NODE_LABELS['DOCUMENT']} {{{PROPERTY_KEYS['DOCUMENT_ID']}: $document_id}})
                      -[:{RELATIONSHIP_TYPES['HAS_SECTION']}]->(s:{NODE_LABELS['SECTION']})
                      -[:{RELATIONSHIP_TYPES['HAS_CHUNK']}]->(c:{NODE_LABELS['CHUNK']})
                WHERE ANY(keyword IN $keywords WHERE toLower(s.{PROPERTY_KEYS['SECTION_TITLE']}) CONTAINS toLower(keyword))
                RETURN c.{PROPERTY_KEYS['CHUNK_ID']} AS chunk_id,
                       c.{PROPERTY_KEYS['CHUNK_INDEX']} AS chunk_index,
                       c.{PROPERTY_KEYS['CHUNK_TYPE']} AS chunk_type,
                       c.{PROPERTY_KEYS['CONTENT']} AS chunk_text,
                       d.{PROPERTY_KEYS['DOCUMENT_ID']} AS document_id,
                       d.{PROPERTY_KEYS['TITLE']} AS filename,
                       s.{PROPERTY_KEYS['SECTION_TITLE']} AS section_title,
                       s.{PROPERTY_KEYS['SECTION_ID']} AS section_id,
                       s.{PROPERTY_KEYS['SECTION_INDEX']} AS section_index
                ORDER BY s.{PROPERTY_KEYS['SECTION_INDEX']}, c.{PROPERTY_KEYS['CHUNK_INDEX']}
                LIMIT $limit
                """
                result = session.run(query, keywords=keywords, document_id=document_id, limit=limit)
            else:
                query = f"""
                MATCH (d:{NODE_LABELS['DOCUMENT']})
                      -[:{RELATIONSHIP_TYPES['HAS_SECTION']}]->(s:{NODE_LABELS['SECTION']})
                      -[:{RELATIONSHIP_TYPES['HAS_CHUNK']}]->(c:{NODE_LABELS['CHUNK']})
                WHERE ANY(keyword IN $keywords WHERE toLower(s.{PROPERTY_KEYS['SECTION_TITLE']}) CONTAINS toLower(keyword))
                RETURN c.{PROPERTY_KEYS['CHUNK_ID']} AS chunk_id,
                       c.{PROPERTY_KEYS['CHUNK_INDEX']} AS chunk_index,
                       c.{PROPERTY_KEYS['CHUNK_TYPE']} AS chunk_type,
                       c.{PROPERTY_KEYS['CONTENT']} AS chunk_text,
                       d.{PROPERTY_KEYS['DOCUMENT_ID']} AS document_id,
                       d.{PROPERTY_KEYS['TITLE']} AS filename,
                       s.{PROPERTY_KEYS['SECTION_TITLE']} AS section_title,
                       s.{PROPERTY_KEYS['SECTION_ID']} AS section_id,
                       s.{PROPERTY_KEYS['SECTION_INDEX']} AS section_index
                ORDER BY s.{PROPERTY_KEYS['SECTION_INDEX']}, c.{PROPERTY_KEYS['CHUNK_INDEX']}
                LIMIT $limit
                """
                result = session.run(query, keywords=keywords, limit=limit)
            
            chunks = []
            for record in result:
                chunk_text = record.get("chunk_text", "")
                chunks.append({
                    "chunk_id": record["chunk_id"],
                    "document_id": record["document_id"],
                    "chunk_index": record["chunk_index"],
                    "chunk_type": record.get("chunk_type", "text"),
                    "chunk_text": chunk_text,
                    "filename": record.get("filename", "Unknown Document"),
                    "graph_score": 0.9,  # High score for section-title matches
                    "metadata": {
                        "section_title": record.get("section_title"),
                        "section_id": record.get("section_id"),
                        "section_index": record.get("section_index"),
                        "retrieval_method": "section_title_match",
                    }
                })
            
            # Log results for debugging
            if chunks:
                sections_found = set(c["metadata"].get("section_title") for c in chunks)
                logger.info(
                    f"Section-title search found {len(chunks)} chunks from {len(sections_found)} sections",
                    extra={"keywords": keywords[:5], "sections": list(sections_found)[:3]}
                )
            else:
                logger.debug(f"No sections found matching keywords: {keywords[:5]}")
            
            # Record metrics
            query_duration = time.time() - query_start
            neo4j_graph_query_duration_seconds.labels(query_type="by_section_title").observe(query_duration)
            neo4j_graph_queries_total.labels(query_type="by_section_title").inc()
            neo4j_chunks_retrieved_via_graph.labels(query_type="by_section_title").inc(len(chunks))
            
            return chunks
        except Exception as e:
            query_duration = time.time() - query_start
            neo4j_graph_queries_total.labels(query_type="by_section_title").inc()
            logger.error(f"Section title query failed: {str(e)}")
            return []
        finally:
            session.close()
    
    def query_by_keywords(
        self,
        keywords: List[str],
        limit: int = 10,
        document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query chunks using a combined strategy:
        1. Find sections whose titles match keywords (primary - gets explanatory content)
        2. Find chunks that mention matching entities (secondary)
        3. Find chunks whose content contains keywords (fallback)
        
        This combined approach ensures we get both:
        - The actual content from relevant sections (e.g., the explanation of Utilitarianism)
        - Related mentions from other parts of the document
        
        Args:
            keywords: List of keywords to search for
            limit: Maximum number of chunks to return
            document_id: Optional document ID to filter results to a specific document
        
        Returns:
            List of chunk dictionaries with content
        """
        query_start = time.time()
        driver = self._get_driver()
        session = driver.session(database=self.database)
        
        all_chunks = []
        seen_chunk_ids = set()
        
        try:
            # Strategy 1: Find chunks from sections with matching titles (PRIMARY)
            # This is critical for getting the actual content about a topic
            section_chunks = self.query_by_section_title(
                keywords=keywords,
                limit=limit,  # Get plenty of section content
                document_id=document_id
            )
            
            for chunk in section_chunks:
                chunk_id = chunk.get("chunk_id")
                if chunk_id and chunk_id not in seen_chunk_ids:
                    all_chunks.append(chunk)
                    seen_chunk_ids.add(chunk_id)
            
            logger.debug(f"Section-title search found {len(section_chunks)} chunks for keywords: {keywords[:5]}")
            
            # Strategy 2: Find chunks through entity mentions
            if len(all_chunks) < limit:
                remaining_limit = limit - len(all_chunks)
                
                if document_id:
                    query = f"""
                    MATCH (d:{NODE_LABELS['DOCUMENT']} {{{PROPERTY_KEYS['DOCUMENT_ID']}: $document_id}})
                          -[:{RELATIONSHIP_TYPES['HAS_SECTION']}]->(s:{NODE_LABELS['SECTION']})
                          -[:{RELATIONSHIP_TYPES['HAS_CHUNK']}]->(c:{NODE_LABELS['CHUNK']})
                          -[:{RELATIONSHIP_TYPES['MENTIONS']}]->(e:{NODE_LABELS['ENTITY']})
                    WHERE ANY(keyword IN $keywords WHERE toLower(e.{PROPERTY_KEYS['ENTITY_NAME']}) CONTAINS toLower(keyword))
                      AND NOT c.{PROPERTY_KEYS['CHUNK_ID']} IN $seen_ids
                    WITH DISTINCT c, d, s, count(DISTINCT e) AS match_count
                    RETURN c.{PROPERTY_KEYS['CHUNK_ID']} AS chunk_id,
                           c.{PROPERTY_KEYS['CHUNK_INDEX']} AS chunk_index,
                           c.{PROPERTY_KEYS['CHUNK_TYPE']} AS chunk_type,
                           c.{PROPERTY_KEYS['CONTENT']} AS chunk_text,
                           d.{PROPERTY_KEYS['DOCUMENT_ID']} AS document_id,
                           d.{PROPERTY_KEYS['TITLE']} AS filename,
                           s.{PROPERTY_KEYS['SECTION_TITLE']} AS section_title,
                           s.{PROPERTY_KEYS['SECTION_INDEX']} AS section_index,
                           match_count
                    ORDER BY match_count DESC, s.{PROPERTY_KEYS['SECTION_INDEX']}, c.{PROPERTY_KEYS['CHUNK_INDEX']}
                    LIMIT $limit
                    """
                    result = session.run(
                        query,
                        keywords=keywords,
                        document_id=document_id,
                        seen_ids=list(seen_chunk_ids),
                        limit=remaining_limit
                    )
                else:
                    query = f"""
                    MATCH (e:{NODE_LABELS['ENTITY']})
                    <-[:{RELATIONSHIP_TYPES['MENTIONS']}]-(c:{NODE_LABELS['CHUNK']})
                    <-[:{RELATIONSHIP_TYPES['HAS_CHUNK']}]-(s:{NODE_LABELS['SECTION']})
                    <-[:{RELATIONSHIP_TYPES['HAS_SECTION']}]-(d:{NODE_LABELS['DOCUMENT']})
                    WHERE ANY(keyword IN $keywords WHERE toLower(e.{PROPERTY_KEYS['ENTITY_NAME']}) CONTAINS toLower(keyword))
                      AND NOT c.{PROPERTY_KEYS['CHUNK_ID']} IN $seen_ids
                    WITH DISTINCT c, d, s, count(DISTINCT e) AS match_count
                    RETURN c.{PROPERTY_KEYS['CHUNK_ID']} AS chunk_id,
                           c.{PROPERTY_KEYS['CHUNK_INDEX']} AS chunk_index,
                           c.{PROPERTY_KEYS['CHUNK_TYPE']} AS chunk_type,
                           c.{PROPERTY_KEYS['CONTENT']} AS chunk_text,
                           d.{PROPERTY_KEYS['DOCUMENT_ID']} AS document_id,
                           d.{PROPERTY_KEYS['TITLE']} AS filename,
                           s.{PROPERTY_KEYS['SECTION_TITLE']} AS section_title,
                           s.{PROPERTY_KEYS['SECTION_INDEX']} AS section_index,
                           match_count
                    ORDER BY match_count DESC, s.{PROPERTY_KEYS['SECTION_INDEX']}, c.{PROPERTY_KEYS['CHUNK_INDEX']}
                    LIMIT $limit
                    """
                    result = session.run(
                        query,
                        keywords=keywords,
                        seen_ids=list(seen_chunk_ids),
                        limit=remaining_limit * 2  # Get extra for diversity
                    )
                
                entity_chunks = []
                for record in result:
                    chunk_id = record["chunk_id"]
                    if chunk_id not in seen_chunk_ids:
                        entity_chunks.append({
                            "chunk_id": chunk_id,
                            "document_id": record["document_id"],
                            "chunk_index": record["chunk_index"],
                            "chunk_type": record.get("chunk_type", "text"),
                            "chunk_text": record.get("chunk_text", ""),
                            "filename": record.get("filename", "Unknown Document"),
                            "match_count": record.get("match_count", 1),
                            "graph_score": 0.7,  # Medium score for entity matches
                            "metadata": {
                                "section_title": record.get("section_title"),
                                "section_index": record.get("section_index"),
                                "retrieval_method": "entity_mention",
                            }
                        })
                        seen_chunk_ids.add(chunk_id)
                
                # Apply document diversity for entity chunks
                if entity_chunks and not document_id:
                    chunks_by_doc = {}
                    for chunk in entity_chunks:
                        doc_id = chunk["document_id"]
                        if doc_id not in chunks_by_doc:
                            chunks_by_doc[doc_id] = []
                        chunks_by_doc[doc_id].append(chunk)
                    
                    # Round-robin across documents
                    doc_ids = list(chunks_by_doc.keys())
                    max_per_doc = max(len(chunks_by_doc[d]) for d in doc_ids) if doc_ids else 0
                    
                    for i in range(max_per_doc):
                        for doc_id in doc_ids:
                            if len(all_chunks) >= limit:
                                break
                            if i < len(chunks_by_doc[doc_id]):
                                all_chunks.append(chunks_by_doc[doc_id][i])
                        if len(all_chunks) >= limit:
                            break
                else:
                    all_chunks.extend(entity_chunks[:remaining_limit])
                
                logger.debug(f"Entity search found {len(entity_chunks)} additional chunks for keywords: {keywords[:5]}")
            
            # Strategy 2.5: Topic-based retrieval for cross-document thematic navigation
            if len(all_chunks) < limit:
                remaining_limit = limit - len(all_chunks)
                topic_chunks = self.query_by_topics(
                    keywords=keywords,
                    limit=remaining_limit,
                    document_id=document_id
                )
                
                topic_added = 0
                for chunk in topic_chunks:
                    chunk_id = chunk.get("chunk_id")
                    if chunk_id and chunk_id not in seen_chunk_ids:
                        all_chunks.append(chunk)
                        seen_chunk_ids.add(chunk_id)
                        topic_added += 1
                
                if topic_added > 0:
                    logger.debug(f"Topic-based search found {topic_added} additional chunks for keywords: {keywords[:5]}")
            
            # Strategy 2.75: Entity relationship traversal for multi-hop discovery
            # This finds chunks related through entity connections (e.g., Equifax -> data breach -> privacy)
            if len(all_chunks) < limit:
                remaining_limit = limit - len(all_chunks)
                # Extract entity names from keywords (assume capitalized words might be entities)
                entity_candidates = [kw for kw in keywords if kw and kw[0].isupper()]
                
                if entity_candidates:
                    try:
                        traversal_chunks = self.traverse_entity_relationships(
                            entity_names=entity_candidates,
                            max_hops=2,
                            limit=remaining_limit
                        )
                        
                        traversal_added = 0
                        for chunk in traversal_chunks:
                            chunk_id = chunk.get("chunk_id")
                            if chunk_id and chunk_id not in seen_chunk_ids:
                                all_chunks.append(chunk)
                                seen_chunk_ids.add(chunk_id)
                                traversal_added += 1
                        
                        if traversal_added > 0:
                            logger.debug(f"Entity traversal found {traversal_added} additional chunks via relationships")
                    except Exception as e:
                        logger.debug(f"Entity traversal skipped: {e}")
            
            # Strategy 3: ALWAYS do content search (not just fallback)
            # Content search finds chunks where the actual text contains keywords
            # This is critical when entity matching finds passing mentions rather than explanatory content
            logger.debug(f"Running content search for keywords: {keywords[:5]}")
            content_chunks = self._query_chunks_by_content(keywords, limit, document_id)
            
            content_added = 0
            for chunk in content_chunks:
                chunk_id = chunk.get("chunk_id")
                if chunk_id and chunk_id not in seen_chunk_ids:
                    chunk["metadata"]["retrieval_method"] = "content_search"
                    # Content search chunks should have decent score
                    if not chunk.get("graph_score"):
                        chunk["graph_score"] = 0.75
                    all_chunks.append(chunk)
                    seen_chunk_ids.add(chunk_id)
                    content_added += 1
            
            if content_added > 0:
                logger.debug(f"Content search added {content_added} additional chunks")
            
            # Sort final results by score
            all_chunks.sort(key=lambda x: (x.get("graph_score", 0.5), -x.get("chunk_index", 0)), reverse=True)
            chunks = all_chunks[:limit]
            
            # Record metrics
            query_duration = time.time() - query_start
            neo4j_graph_query_duration_seconds.labels(query_type="by_keywords").observe(query_duration)
            neo4j_graph_queries_total.labels(query_type="by_keywords").inc()
            neo4j_chunks_retrieved_via_graph.labels(query_type="by_keywords").inc(len(chunks))
            neo4j_query_duration_seconds.labels(operation="read").observe(query_duration)
            neo4j_queries_total.labels(operation="read", status="success").inc()
            
            if chunks:
                # Log retrieval summary for debugging
                methods = {}
                for c in chunks:
                    method = c.get("metadata", {}).get("retrieval_method", "unknown")
                    methods[method] = methods.get(method, 0) + 1
                
                content_found = sum(1 for c in chunks if c.get("chunk_text", "").strip())
                logger.info(
                    f"Keyword search completed: {len(chunks)} chunks ({content_found} with content), "
                    f"methods: {methods}"
                )
            else:
                logger.warning(f"No chunks found for keywords: {keywords[:5]}")
            
            return chunks
        except Exception as e:
            query_duration = time.time() - query_start
            neo4j_queries_total.labels(operation="read", status="error").inc()
            logger.error(f"Graph query by keywords failed: {str(e)}")
            return []
        finally:
            session.close()
    
    def _query_chunks_by_content(
        self,
        keywords: List[str],
        limit: int = 10,
        document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query chunks by searching for keywords directly in chunk content.
        
        This is CRITICAL for finding relevant content when:
        - Entity extraction missed important terms
        - Section titles are generic (e.g., "Document Content")
        - Keywords appear in explanatory text, not as formal entities
        
        Uses multiple keyword matching to prioritize chunks that match more query terms.
        
        Args:
            keywords: List of keywords to search for in chunk content
            limit: Maximum number of chunks to return
            document_id: Optional document ID to filter results
        
        Returns:
            List of chunk dictionaries with relevance scores
        """
        query_start = time.time()
        driver = self._get_driver()
        session = driver.session(database=self.database)
        try:
            if document_id:
                # Improved query: Count keyword matches for better ranking
                query = f"""
                MATCH (d:{NODE_LABELS['DOCUMENT']} {{{PROPERTY_KEYS['DOCUMENT_ID']}: $document_id}})
                      -[:{RELATIONSHIP_TYPES['HAS_SECTION']}]->(s:{NODE_LABELS['SECTION']})
                      -[:{RELATIONSHIP_TYPES['HAS_CHUNK']}]->(c:{NODE_LABELS['CHUNK']})
                WITH c, d, s,
                     size([keyword IN $keywords WHERE toLower(c.{PROPERTY_KEYS['CONTENT']}) CONTAINS toLower(keyword)]) AS match_count
                WHERE match_count > 0
                RETURN DISTINCT c.{PROPERTY_KEYS['CHUNK_ID']} AS chunk_id,
                       c.{PROPERTY_KEYS['CHUNK_INDEX']} AS chunk_index,
                       c.{PROPERTY_KEYS['CHUNK_TYPE']} AS chunk_type,
                       c.{PROPERTY_KEYS['CONTENT']} AS chunk_text,
                       d.{PROPERTY_KEYS['DOCUMENT_ID']} AS document_id,
                       d.{PROPERTY_KEYS['TITLE']} AS filename,
                       s.{PROPERTY_KEYS['SECTION_TITLE']} AS section_title,
                       s.{PROPERTY_KEYS['SECTION_INDEX']} AS section_index,
                       match_count
                ORDER BY match_count DESC, s.{PROPERTY_KEYS['SECTION_INDEX']}, c.{PROPERTY_KEYS['CHUNK_INDEX']}
                LIMIT $limit
                """
                result = session.run(query, keywords=keywords, document_id=document_id, limit=limit)
            else:
                # Improved query: Count how many keywords match in each chunk
                # Prioritize chunks that match more keywords
                query = f"""
                MATCH (d:{NODE_LABELS['DOCUMENT']})
                      -[:{RELATIONSHIP_TYPES['HAS_SECTION']}]->(s:{NODE_LABELS['SECTION']})
                      -[:{RELATIONSHIP_TYPES['HAS_CHUNK']}]->(c:{NODE_LABELS['CHUNK']})
                WITH c, d, s, 
                     size([keyword IN $keywords WHERE toLower(c.{PROPERTY_KEYS['CONTENT']}) CONTAINS toLower(keyword)]) AS match_count
                WHERE match_count > 0
                RETURN c.{PROPERTY_KEYS['CHUNK_ID']} AS chunk_id,
                       c.{PROPERTY_KEYS['CHUNK_INDEX']} AS chunk_index,
                       c.{PROPERTY_KEYS['CHUNK_TYPE']} AS chunk_type,
                       c.{PROPERTY_KEYS['CONTENT']} AS chunk_text,
                       d.{PROPERTY_KEYS['DOCUMENT_ID']} AS document_id,
                       d.{PROPERTY_KEYS['TITLE']} AS filename,
                       s.{PROPERTY_KEYS['SECTION_TITLE']} AS section_title,
                       s.{PROPERTY_KEYS['SECTION_INDEX']} AS section_index,
                       match_count
                ORDER BY match_count DESC, s.{PROPERTY_KEYS['SECTION_INDEX']}, c.{PROPERTY_KEYS['CHUNK_INDEX']}
                LIMIT $limit * 2
                """
                result = session.run(query, keywords=keywords, limit=limit * 2)
            
            chunks = []
            for record in result:
                match_count = record.get("match_count", 1)
                chunk_text = record.get("chunk_text", "").lower()
                
                # Calculate a simple relevance score based on:
                # 1. Number of keywords matched (higher is better)
                # 2. Keyword frequency in chunk (more mentions = more relevant)
                keyword_frequency = sum(chunk_text.count(keyword.lower()) for keyword in keywords)
                relevance_score = match_count * 1.0 + keyword_frequency * 0.1
                
                chunks.append({
                    "chunk_id": record["chunk_id"],
                    "document_id": record["document_id"],
                    "chunk_index": record["chunk_index"],
                    "chunk_type": record.get("chunk_type", "text"),
                    "chunk_text": record.get("chunk_text", ""),
                    "filename": record.get("filename", "Unknown Document"),
                    "graph_score": relevance_score / max(len(keywords), 1),  # Normalize score
                    "metadata": {
                        "section_title": record.get("section_title"),
                        "section_index": record.get("section_index"),
                        "match_count": match_count,
                        "keyword_frequency": keyword_frequency,
                    }
                })
            
            # Sort by relevance score (highest first) and limit
            chunks.sort(key=lambda x: x.get("graph_score", 0), reverse=True)
            chunks = chunks[:limit]
            
            query_duration = time.time() - query_start
            if chunks:
                logger.info(f"Fallback content search found {len(chunks)} chunks matching keywords in content: {keywords[:5]}")
            
            return chunks
        finally:
            session.close()
    
    def query_by_document(
        self,
        document_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Query all chunks from a specific document.
        
        Args:
            document_id: Document ID (UUID as string)
            limit: Maximum number of chunks to return
        
        Returns:
            List of chunk dictionaries
        """
        query = f"""
        MATCH (d:{NODE_LABELS['DOCUMENT']} {{{PROPERTY_KEYS['DOCUMENT_ID']}: $document_id}})
              -[:{RELATIONSHIP_TYPES['HAS_SECTION']}]->(s:{NODE_LABELS['SECTION']})
              -[:{RELATIONSHIP_TYPES['HAS_CHUNK']}]->(c:{NODE_LABELS['CHUNK']})
        RETURN c.{PROPERTY_KEYS['CHUNK_ID']} AS chunk_id,
               c.{PROPERTY_KEYS['CHUNK_INDEX']} AS chunk_index,
               c.{PROPERTY_KEYS['CHUNK_TYPE']} AS chunk_type,
               c.{PROPERTY_KEYS['CONTENT']} AS chunk_text,
               d.{PROPERTY_KEYS['DOCUMENT_ID']} AS document_id,
               d.{PROPERTY_KEYS['TITLE']} AS filename,
               s.{PROPERTY_KEYS['SECTION_TITLE']} AS section_title,
               s.{PROPERTY_KEYS['SECTION_INDEX']} AS section_index
        ORDER BY s.{PROPERTY_KEYS['SECTION_INDEX']}, c.{PROPERTY_KEYS['CHUNK_INDEX']}
        LIMIT $limit
        """
        
        query_start = time.time()
        driver = self._get_driver()
        session = driver.session(database=self.database)
        try:
            result = session.run(query, document_id=document_id, limit=limit)
            chunks = []
            for record in result:
                chunks.append({
                    "chunk_id": record["chunk_id"],
                    "document_id": record["document_id"],
                    "chunk_index": record["chunk_index"],
                    "chunk_type": record.get("chunk_type", "text"),
                    "chunk_text": record.get("chunk_text", ""),  # Content from Neo4j
                    "filename": record.get("filename", "Unknown Document"),
                    "metadata": {
                        "section_title": record.get("section_title"),
                        "section_index": record.get("section_index"),
                    }
                })
            
            # Record metrics
            query_duration = time.time() - query_start
            neo4j_graph_query_duration_seconds.labels(query_type="by_document").observe(query_duration)
            neo4j_graph_queries_total.labels(query_type="by_document").inc()
            neo4j_chunks_retrieved_via_graph.labels(query_type="by_document").inc(len(chunks))
            neo4j_query_duration_seconds.labels(operation="read").observe(query_duration)
            neo4j_queries_total.labels(operation="read", status="success").inc()
            
            return chunks
        except Exception as e:
            query_duration = time.time() - query_start
            neo4j_graph_queries_total.labels(query_type="by_document").inc()
            neo4j_query_duration_seconds.labels(operation="read").observe(query_duration)
            neo4j_queries_total.labels(operation="read", status="error").inc()
            logger.error(f"Graph query by document failed: {str(e)}")
            raise
        finally:
            session.close()
    
    def query_with_graph_traversal(
        self,
        initial_chunk_ids: List[str],
        hop_count: int = 2,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Multi-hop graph traversal from initial chunk hits.
        
        This implements the "hop across the graph" strategy:
        - Start with initial chunks (from entity/keyword matches)
        - Follow edges to find related chunks through shared entities
        - Explore graph neighborhood for richer context
        
        Args:
            initial_chunk_ids: List of starting chunk IDs
            hop_count: Number of hops to traverse (default: 2)
            limit: Maximum number of chunks to return
        
        Returns:
            List of related chunk dictionaries
        """
        if not initial_chunk_ids:
            return []
        
        query = f"""
        MATCH path = (start:{NODE_LABELS['CHUNK']})
              -[:{RELATIONSHIP_TYPES['MENTIONS']}]->(e:{NODE_LABELS['ENTITY']})
              <-[:{RELATIONSHIP_TYPES['MENTIONS']}]-(related:{NODE_LABELS['CHUNK']})
              <-[:{RELATIONSHIP_TYPES['HAS_CHUNK']}]-(s:{NODE_LABELS['SECTION']})
              <-[:{RELATIONSHIP_TYPES['HAS_SECTION']}]-(d:{NODE_LABELS['DOCUMENT']})
        WHERE start.{PROPERTY_KEYS['CHUNK_ID']} IN $chunk_ids
          AND related.{PROPERTY_KEYS['CHUNK_ID']} <> start.{PROPERTY_KEYS['CHUNK_ID']}
        WITH DISTINCT related, d, s, count(DISTINCT e) AS shared_entities
        RETURN related.{PROPERTY_KEYS['CHUNK_ID']} AS chunk_id,
               related.{PROPERTY_KEYS['CHUNK_INDEX']} AS chunk_index,
               related.{PROPERTY_KEYS['CHUNK_TYPE']} AS chunk_type,
               related.{PROPERTY_KEYS['CONTENT']} AS chunk_text,
               d.{PROPERTY_KEYS['DOCUMENT_ID']} AS document_id,
               d.{PROPERTY_KEYS['TITLE']} AS filename,
               s.{PROPERTY_KEYS['SECTION_TITLE']} AS section_title,
               s.{PROPERTY_KEYS['SECTION_INDEX']} AS section_index,
               shared_entities
        ORDER BY shared_entities DESC, s.{PROPERTY_KEYS['SECTION_INDEX']}, related.{PROPERTY_KEYS['CHUNK_INDEX']}
        LIMIT $limit
        """
        
        query_start = time.time()
        driver = self._get_driver()
        session = driver.session(database=self.database)
        try:
            result = session.run(query, chunk_ids=initial_chunk_ids, limit=limit)
            chunks = []
            for record in result:
                chunks.append({
                    "chunk_id": record["chunk_id"],
                    "document_id": record["document_id"],
                    "chunk_index": record["chunk_index"],
                    "chunk_type": record.get("chunk_type", "text"),
                    "chunk_text": record.get("chunk_text", ""),
                    "filename": record.get("filename", "Unknown Document"),
                    "metadata": {
                        "section_title": record.get("section_title"),
                        "section_index": record.get("section_index"),
                        "shared_entities": record.get("shared_entities", 0),
                        "via_traversal": True,  # Mark as graph traversal result
                    }
                })
            
            query_duration = time.time() - query_start
            neo4j_graph_query_duration_seconds.labels(query_type="graph_traversal").observe(query_duration)
            neo4j_graph_queries_total.labels(query_type="graph_traversal").inc()
            neo4j_chunks_retrieved_via_graph.labels(query_type="graph_traversal").inc(len(chunks))
            
            return chunks
        finally:
            session.close()
    
    def query_chunks_by_section_id(
        self,
        section_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get ALL chunks from a specific section.
        
        Useful for context expansion - when you find a relevant section,
        this retrieves all its content chunks in order.
        
        Args:
            section_id: Section identifier
            limit: Maximum chunks to return
        
        Returns:
            List of chunk dictionaries in order
        """
        query = f"""
        MATCH (s:{NODE_LABELS['SECTION']} {{{PROPERTY_KEYS['SECTION_ID']}: $section_id}})
              <-[:{RELATIONSHIP_TYPES['HAS_SECTION']}]-(d:{NODE_LABELS['DOCUMENT']})
        MATCH (s)-[:{RELATIONSHIP_TYPES['HAS_CHUNK']}]->(c:{NODE_LABELS['CHUNK']})
        RETURN c.{PROPERTY_KEYS['CHUNK_ID']} AS chunk_id,
               c.{PROPERTY_KEYS['CHUNK_INDEX']} AS chunk_index,
               c.{PROPERTY_KEYS['CHUNK_TYPE']} AS chunk_type,
               c.{PROPERTY_KEYS['CONTENT']} AS chunk_text,
               d.{PROPERTY_KEYS['DOCUMENT_ID']} AS document_id,
               d.{PROPERTY_KEYS['TITLE']} AS filename,
               s.{PROPERTY_KEYS['SECTION_TITLE']} AS section_title,
               s.{PROPERTY_KEYS['SECTION_INDEX']} AS section_index
        ORDER BY c.{PROPERTY_KEYS['CHUNK_INDEX']}
        LIMIT $limit
        """
        
        query_start = time.time()
        driver = self._get_driver()
        session = driver.session(database=self.database)
        try:
            result = session.run(query, section_id=section_id, limit=limit)
            chunks = []
            for record in result:
                chunks.append({
                    "chunk_id": record["chunk_id"],
                    "document_id": record["document_id"],
                    "chunk_index": record["chunk_index"],
                    "chunk_type": record.get("chunk_type", "text"),
                    "chunk_text": record.get("chunk_text", ""),
                    "filename": record.get("filename", "Unknown Document"),
                    "metadata": {
                        "section_title": record.get("section_title"),
                        "section_index": record.get("section_index"),
                    }
                })
            
            query_duration = time.time() - query_start
            neo4j_graph_query_duration_seconds.labels(query_type="by_section_id").observe(query_duration)
            neo4j_graph_queries_total.labels(query_type="by_section_id").inc()
            neo4j_chunks_retrieved_via_graph.labels(query_type="by_section_id").inc(len(chunks))
            
            return chunks
        except Exception as e:
            logger.error(f"Query by section_id failed: {str(e)}")
            return []
        finally:
            session.close()
    
    def query_related_chunks(
        self,
        chunk_id: str,
        hop_count: int = 2,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Query chunks related to a given chunk through entity relationships.
        
        This enables multi-hop queries: find chunks that share entities
        with the given chunk.
        
        Args:
            chunk_id: Starting chunk ID (UUID as string)
            hop_count: Number of hops to traverse (default: 2)
            limit: Maximum number of chunks to return
        
        Returns:
            List of related chunk dictionaries
        """
        query = f"""
        MATCH (start:{NODE_LABELS['CHUNK']} {{{PROPERTY_KEYS['CHUNK_ID']}: $chunk_id}})
              -[:{RELATIONSHIP_TYPES['MENTIONS']}]->(e:{NODE_LABELS['ENTITY']})
              <-[:{RELATIONSHIP_TYPES['MENTIONS']}]-(related:{NODE_LABELS['CHUNK']})
              <-[:{RELATIONSHIP_TYPES['HAS_CHUNK']}]-(s:{NODE_LABELS['SECTION']})
              <-[:{RELATIONSHIP_TYPES['HAS_SECTION']}]-(d:{NODE_LABELS['DOCUMENT']})
        WHERE related.{PROPERTY_KEYS['CHUNK_ID']} <> $chunk_id
        RETURN DISTINCT related.{PROPERTY_KEYS['CHUNK_ID']} AS chunk_id,
               related.{PROPERTY_KEYS['CHUNK_INDEX']} AS chunk_index,
               related.{PROPERTY_KEYS['CHUNK_TYPE']} AS chunk_type,
               related.{PROPERTY_KEYS['CONTENT']} AS chunk_text,
               d.{PROPERTY_KEYS['DOCUMENT_ID']} AS document_id,
               d.{PROPERTY_KEYS['TITLE']} AS filename,
               s.{PROPERTY_KEYS['SECTION_TITLE']} AS section_title,
               s.{PROPERTY_KEYS['SECTION_INDEX']} AS section_index,
               count(e) AS shared_entities
        ORDER BY shared_entities DESC, s.{PROPERTY_KEYS['SECTION_INDEX']}, related.{PROPERTY_KEYS['CHUNK_INDEX']}
        LIMIT $limit
        """
        
        query_start = time.time()
        driver = self._get_driver()
        session = driver.session(database=self.database)
        try:
            result = session.run(query, chunk_id=chunk_id, limit=limit)
            chunks = []
            for record in result:
                chunks.append({
                    "chunk_id": record["chunk_id"],
                    "document_id": record["document_id"],
                    "chunk_index": record["chunk_index"],
                    "chunk_type": record.get("chunk_type", "text"),
                    "chunk_text": record.get("chunk_text", ""),  # Content from Neo4j
                    "filename": record.get("filename", "Unknown Document"),
                    "metadata": {
                        "section_title": record.get("section_title"),
                        "section_index": record.get("section_index"),
                        "shared_entities": record.get("shared_entities", 0),
                    }
                })
            
            # Record metrics
            query_duration = time.time() - query_start
            neo4j_graph_query_duration_seconds.labels(query_type="related_chunks").observe(query_duration)
            neo4j_graph_queries_total.labels(query_type="related_chunks").inc()
            neo4j_chunks_retrieved_via_graph.labels(query_type="related_chunks").inc(len(chunks))
            neo4j_query_duration_seconds.labels(operation="read").observe(query_duration)
            neo4j_queries_total.labels(operation="read", status="success").inc()
            
            return chunks
        except Exception as e:
            query_duration = time.time() - query_start
            neo4j_graph_queries_total.labels(query_type="related_chunks").inc()
            neo4j_query_duration_seconds.labels(operation="read").observe(query_duration)
            neo4j_queries_total.labels(operation="read", status="error").inc()
            logger.error(f"Graph query for related chunks failed: {str(e)}")
            raise
        finally:
            session.close()
    
    def query_with_context_expansion(
        self,
        chunk_ids: List[str],
        expand_before: int = 1,
        expand_after: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Expand initial chunks with their sequential neighbors using NEXT_CHUNK.
        
        Given a set of retrieved chunks, this finds their immediate neighbors
        (previous/next chunks in the document) for richer context.
        
        This is essential for providing full context to the LLM - a single chunk
        might cut off mid-sentence, so including neighbors helps.
        
        Args:
            chunk_ids: List of initial chunk IDs to expand
            expand_before: Number of previous chunks to include (default: 1)
            expand_after: Number of next chunks to include (default: 1)
        
        Returns:
            List of expanded chunk dictionaries (includes originals + neighbors)
        """
        if not chunk_ids:
            return []
        
        # Query to get chunks and their neighbors via NEXT_CHUNK
        query = f"""
        UNWIND $chunk_ids AS cid
        MATCH (c:{NODE_LABELS['CHUNK']} {{{PROPERTY_KEYS['CHUNK_ID']}: cid}})
              <-[:{RELATIONSHIP_TYPES['HAS_CHUNK']}]-(s:{NODE_LABELS['SECTION']})
              <-[:{RELATIONSHIP_TYPES['HAS_SECTION']}]-(d:{NODE_LABELS['DOCUMENT']})
        
        // Get the chunk itself
        WITH c, s, d, cid, 0 AS distance
        
        // Get previous chunks (following NEXT_CHUNK backwards)
        OPTIONAL MATCH (prev:{NODE_LABELS['CHUNK']})-[:{RELATIONSHIP_TYPES['NEXT_CHUNK']}*1..{expand_before}]->(c)
        
        // Get next chunks (following NEXT_CHUNK forwards)
        OPTIONAL MATCH (c)-[:{RELATIONSHIP_TYPES['NEXT_CHUNK']}*1..{expand_after}]->(next:{NODE_LABELS['CHUNK']})
        
        // Collect all chunks
        WITH d, s, collect(DISTINCT c) + collect(DISTINCT prev) + collect(DISTINCT next) AS all_chunks
        UNWIND all_chunks AS chunk
        WHERE chunk IS NOT NULL
        
        RETURN DISTINCT chunk.{PROPERTY_KEYS['CHUNK_ID']} AS chunk_id,
               chunk.{PROPERTY_KEYS['CHUNK_INDEX']} AS chunk_index,
               chunk.{PROPERTY_KEYS['CHUNK_TYPE']} AS chunk_type,
               chunk.{PROPERTY_KEYS['CONTENT']} AS chunk_text,
               d.{PROPERTY_KEYS['DOCUMENT_ID']} AS document_id,
               d.{PROPERTY_KEYS['TITLE']} AS filename,
               s.{PROPERTY_KEYS['SECTION_TITLE']} AS section_title,
               s.{PROPERTY_KEYS['SECTION_INDEX']} AS section_index
        ORDER BY d.{PROPERTY_KEYS['DOCUMENT_ID']}, s.{PROPERTY_KEYS['SECTION_INDEX']}, chunk.{PROPERTY_KEYS['CHUNK_INDEX']}
        """
        
        query_start = time.time()
        driver = self._get_driver()
        session = driver.session(database=self.database)
        try:
            result = session.run(query, chunk_ids=chunk_ids)
            chunks = []
            seen_ids = set()
            
            for record in result:
                chunk_id = record["chunk_id"]
                if chunk_id and chunk_id not in seen_ids:
                    chunks.append({
                        "chunk_id": chunk_id,
                        "document_id": record["document_id"],
                        "chunk_index": record["chunk_index"],
                        "chunk_type": record.get("chunk_type", "text"),
                        "chunk_text": record.get("chunk_text", ""),
                        "filename": record.get("filename", "Unknown Document"),
                        "metadata": {
                            "section_title": record.get("section_title"),
                            "section_index": record.get("section_index"),
                            "is_context_expansion": chunk_id not in chunk_ids,
                        }
                    })
                    seen_ids.add(chunk_id)
            
            query_duration = time.time() - query_start
            neo4j_graph_query_duration_seconds.labels(query_type="context_expansion").observe(query_duration)
            neo4j_graph_queries_total.labels(query_type="context_expansion").inc()
            
            original_count = len([c for c in chunks if c["chunk_id"] in chunk_ids])
            expanded_count = len(chunks) - original_count
            logger.debug(f"Context expansion: {original_count} original + {expanded_count} neighbors = {len(chunks)} total")
            
            return chunks
        except Exception as e:
            logger.error(f"Context expansion query failed: {str(e)}")
            return []
        finally:
            session.close()
    
    def query_cross_document_entities(
        self,
        entity_names: List[str],
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Find all documents and chunks that mention the same entities.
        
        Uses normalized_name for cross-document entity resolution.
        This enables discovering related content across different documents.
        
        Args:
            entity_names: List of entity names to search for
            limit: Maximum number of chunks to return
        
        Returns:
            List of chunk dictionaries from multiple documents
        """
        if not entity_names:
            return []
        
        # Normalize entity names for matching
        normalized_names = [name.lower().strip() for name in entity_names]
        
        query = f"""
        UNWIND $normalized_names AS norm_name
        MATCH (e:{NODE_LABELS['ENTITY']} {{{PROPERTY_KEYS['ENTITY_NORMALIZED_NAME']}: norm_name}})
              <-[:{RELATIONSHIP_TYPES['MENTIONS']}]-(c:{NODE_LABELS['CHUNK']})
              <-[:{RELATIONSHIP_TYPES['HAS_CHUNK']}]-(s:{NODE_LABELS['SECTION']})
              <-[:{RELATIONSHIP_TYPES['HAS_SECTION']}]-(d:{NODE_LABELS['DOCUMENT']})
        
        WITH c, s, d, e, count(DISTINCT e) AS entity_matches
        
        RETURN DISTINCT c.{PROPERTY_KEYS['CHUNK_ID']} AS chunk_id,
               c.{PROPERTY_KEYS['CHUNK_INDEX']} AS chunk_index,
               c.{PROPERTY_KEYS['CHUNK_TYPE']} AS chunk_type,
               c.{PROPERTY_KEYS['CONTENT']} AS chunk_text,
               d.{PROPERTY_KEYS['DOCUMENT_ID']} AS document_id,
               d.{PROPERTY_KEYS['TITLE']} AS filename,
               s.{PROPERTY_KEYS['SECTION_TITLE']} AS section_title,
               s.{PROPERTY_KEYS['SECTION_INDEX']} AS section_index,
               entity_matches
        ORDER BY entity_matches DESC, d.{PROPERTY_KEYS['DOCUMENT_ID']}, c.{PROPERTY_KEYS['CHUNK_INDEX']}
        LIMIT $limit
        """
        
        query_start = time.time()
        driver = self._get_driver()
        session = driver.session(database=self.database)
        try:
            result = session.run(query, normalized_names=normalized_names, limit=limit)
            chunks = []
            documents_found = set()
            
            for record in result:
                chunk_id = record["chunk_id"]
                doc_id = record["document_id"]
                documents_found.add(doc_id)
                
                chunks.append({
                    "chunk_id": chunk_id,
                    "document_id": doc_id,
                    "chunk_index": record["chunk_index"],
                    "chunk_type": record.get("chunk_type", "text"),
                    "chunk_text": record.get("chunk_text", ""),
                    "filename": record.get("filename", "Unknown Document"),
                    "graph_score": 0.85,
                    "metadata": {
                        "section_title": record.get("section_title"),
                        "section_index": record.get("section_index"),
                        "entity_matches": record.get("entity_matches", 1),
                        "retrieval_method": "cross_document_entity",
                    }
                })
            
            query_duration = time.time() - query_start
            neo4j_graph_query_duration_seconds.labels(query_type="cross_document_entities").observe(query_duration)
            neo4j_graph_queries_total.labels(query_type="cross_document_entities").inc()
            neo4j_chunks_retrieved_via_graph.labels(query_type="cross_document_entities").inc(len(chunks))
            
            logger.info(
                f"Cross-document entity search: found {len(chunks)} chunks across {len(documents_found)} documents "
                f"for entities: {entity_names[:3]}..."
            )
            
            return chunks
        except Exception as e:
            logger.error(f"Cross-document entity query failed: {str(e)}")
            return []
        finally:
            session.close()
    
    def query_by_topics(
        self,
        keywords: List[str],
        limit: int = 10,
        document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query chunks through the topic layer for cross-document thematic navigation.
        
        This enables finding content across documents that share the same topics,
        even if they don't share exact keywords or entities.
        
        Strategy:
        1. Find topics matching the keywords
        2. Find chunks linked to those topics
        3. Return chunks from relevant topic neighborhoods
        
        Args:
            keywords: List of keywords to match against topic names/keywords
            limit: Maximum number of chunks to return
            document_id: Optional document ID to filter results
            
        Returns:
            List of chunk dictionaries from topic-based retrieval
        """
        query_start = time.time()
        driver = self._get_driver()
        session = driver.session(database=self.database)
        
        try:
            # Build query to find topics and their associated chunks
            if document_id:
                query = f"""
                MATCH (t:{NODE_LABELS['TOPIC']})
                WHERE ANY(keyword IN $keywords WHERE 
                    toLower(t.{PROPERTY_KEYS['TOPIC_NAME']}) CONTAINS toLower(keyword) OR
                    ANY(kw IN t.{PROPERTY_KEYS['TOPIC_KEYWORDS']} WHERE toLower(kw) CONTAINS toLower(keyword))
                )
                WITH t
                MATCH (d:{NODE_LABELS['DOCUMENT']} {{{PROPERTY_KEYS['DOCUMENT_ID']}: $document_id}})
                      -[:{RELATIONSHIP_TYPES['HAS_SECTION']}]->(s:{NODE_LABELS['SECTION']})
                      -[:{RELATIONSHIP_TYPES['HAS_CHUNK']}]->(c:{NODE_LABELS['CHUNK']})
                      -[r:{RELATIONSHIP_TYPES['HAS_TOPIC']}]->(t)
                WITH DISTINCT c, d, s, t, r.relevance AS relevance
                RETURN c.{PROPERTY_KEYS['CHUNK_ID']} AS chunk_id,
                       c.{PROPERTY_KEYS['CHUNK_INDEX']} AS chunk_index,
                       c.{PROPERTY_KEYS['CHUNK_TYPE']} AS chunk_type,
                       c.{PROPERTY_KEYS['CONTENT']} AS chunk_text,
                       d.{PROPERTY_KEYS['DOCUMENT_ID']} AS document_id,
                       d.{PROPERTY_KEYS['TITLE']} AS filename,
                       s.{PROPERTY_KEYS['SECTION_TITLE']} AS section_title,
                       s.{PROPERTY_KEYS['SECTION_INDEX']} AS section_index,
                       t.{PROPERTY_KEYS['TOPIC_NAME']} AS topic_name,
                       COALESCE(relevance, 0.5) AS relevance
                ORDER BY relevance DESC, s.{PROPERTY_KEYS['SECTION_INDEX']}, c.{PROPERTY_KEYS['CHUNK_INDEX']}
                LIMIT $limit
                """
                result = session.run(query, keywords=keywords, document_id=document_id, limit=limit)
            else:
                # Cross-document topic-based retrieval
                query = f"""
                MATCH (t:{NODE_LABELS['TOPIC']})
                WHERE ANY(keyword IN $keywords WHERE 
                    toLower(t.{PROPERTY_KEYS['TOPIC_NAME']}) CONTAINS toLower(keyword) OR
                    ANY(kw IN t.{PROPERTY_KEYS['TOPIC_KEYWORDS']} WHERE toLower(kw) CONTAINS toLower(keyword))
                )
                WITH t
                MATCH (c:{NODE_LABELS['CHUNK']})
                      -[r:{RELATIONSHIP_TYPES['HAS_TOPIC']}]->(t)
                MATCH (s:{NODE_LABELS['SECTION']})
                      -[:{RELATIONSHIP_TYPES['HAS_CHUNK']}]->(c)
                MATCH (d:{NODE_LABELS['DOCUMENT']})
                      -[:{RELATIONSHIP_TYPES['HAS_SECTION']}]->(s)
                WITH DISTINCT c, d, s, t, r.relevance AS relevance
                RETURN c.{PROPERTY_KEYS['CHUNK_ID']} AS chunk_id,
                       c.{PROPERTY_KEYS['CHUNK_INDEX']} AS chunk_index,
                       c.{PROPERTY_KEYS['CHUNK_TYPE']} AS chunk_type,
                       c.{PROPERTY_KEYS['CONTENT']} AS chunk_text,
                       d.{PROPERTY_KEYS['DOCUMENT_ID']} AS document_id,
                       d.{PROPERTY_KEYS['TITLE']} AS filename,
                       s.{PROPERTY_KEYS['SECTION_TITLE']} AS section_title,
                       s.{PROPERTY_KEYS['SECTION_INDEX']} AS section_index,
                       t.{PROPERTY_KEYS['TOPIC_NAME']} AS topic_name,
                       COALESCE(relevance, 0.5) AS relevance
                ORDER BY relevance DESC, d.{PROPERTY_KEYS['DOCUMENT_ID']}, c.{PROPERTY_KEYS['CHUNK_INDEX']}
                LIMIT $limit
                """
                result = session.run(query, keywords=keywords, limit=limit * 2)
            
            chunks = []
            topics_found = set()
            
            for record in result:
                chunk_id = record["chunk_id"]
                topic_name = record.get("topic_name", "Unknown")
                topics_found.add(topic_name)
                
                chunks.append({
                    "chunk_id": chunk_id,
                    "document_id": record["document_id"],
                    "chunk_index": record["chunk_index"],
                    "chunk_type": record.get("chunk_type", "text"),
                    "chunk_text": record.get("chunk_text", ""),
                    "filename": record.get("filename", "Unknown Document"),
                    "graph_score": 0.75 + (record.get("relevance", 0.5) * 0.15),  # 0.75-0.9 range
                    "metadata": {
                        "section_title": record.get("section_title"),
                        "section_index": record.get("section_index"),
                        "topic": topic_name,
                        "retrieval_method": "topic_based",
                    }
                })
            
            query_duration = time.time() - query_start
            neo4j_graph_query_duration_seconds.labels(query_type="by_topics").observe(query_duration)
            neo4j_graph_queries_total.labels(query_type="by_topics").inc()
            neo4j_chunks_retrieved_via_graph.labels(query_type="by_topics").inc(len(chunks))
            
            if chunks:
                logger.info(
                    f"Topic-based search: found {len(chunks)} chunks across {len(topics_found)} topics "
                    f"for keywords: {keywords[:5]}"
                )
            
            return chunks[:limit]
        except Exception as e:
            logger.error(f"Topic-based query failed: {str(e)}")
            return []
        finally:
            session.close()
    
    def traverse_entity_relationships(
        self,
        entity_names: List[str],
        max_hops: int = 2,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Traverse entity relationships to find related entities and their chunks.
        
        This enables multi-hop discovery:
        - Start with initial entities (e.g., "Equifax")
        - Follow RELATED_TO edges to find connected entities (e.g., "data breach", "privacy")
        - Find chunks mentioning those related entities
        
        Args:
            entity_names: Starting entity names
            max_hops: Maximum number of relationship hops (default: 2)
            limit: Maximum number of chunks to return
            
        Returns:
            List of chunk dictionaries from related entities
        """
        query_start = time.time()
        driver = self._get_driver()
        session = driver.session(database=self.database)
        
        try:
            # Normalize entity names
            normalized_names = [name.lower().strip() for name in entity_names]
            
            # Query to traverse entity relationships
            query = f"""
            MATCH path = (e1:{NODE_LABELS['ENTITY']})
                         -[:{RELATIONSHIP_TYPES['RELATED_TO']}*1..{max_hops}]-
                         (e2:{NODE_LABELS['ENTITY']})
            WHERE e1.{PROPERTY_KEYS['ENTITY_NORMALIZED_NAME']} IN $normalized_names
            WITH DISTINCT e2, length(path) AS hops
            MATCH (c:{NODE_LABELS['CHUNK']})
                  -[:{RELATIONSHIP_TYPES['MENTIONS']}]->(e2)
            MATCH (s:{NODE_LABELS['SECTION']})
                  -[:{RELATIONSHIP_TYPES['HAS_CHUNK']}]->(c)
            MATCH (d:{NODE_LABELS['DOCUMENT']})
                  -[:{RELATIONSHIP_TYPES['HAS_SECTION']}]->(s)
            WITH DISTINCT c, d, s, e2, hops
            RETURN c.{PROPERTY_KEYS['CHUNK_ID']} AS chunk_id,
                   c.{PROPERTY_KEYS['CHUNK_INDEX']} AS chunk_index,
                   c.{PROPERTY_KEYS['CHUNK_TYPE']} AS chunk_type,
                   c.{PROPERTY_KEYS['CONTENT']} AS chunk_text,
                   d.{PROPERTY_KEYS['DOCUMENT_ID']} AS document_id,
                   d.{PROPERTY_KEYS['TITLE']} AS filename,
                   s.{PROPERTY_KEYS['SECTION_TITLE']} AS section_title,
                   s.{PROPERTY_KEYS['SECTION_INDEX']} AS section_index,
                   e2.{PROPERTY_KEYS['ENTITY_NAME']} AS related_entity,
                   hops
            ORDER BY hops ASC, d.{PROPERTY_KEYS['DOCUMENT_ID']}, c.{PROPERTY_KEYS['CHUNK_INDEX']}
            LIMIT $limit
            """
            
            result = session.run(query, normalized_names=normalized_names, limit=limit)
            chunks = []
            related_entities = set()
            
            for record in result:
                chunk_id = record["chunk_id"]
                related_entity = record.get("related_entity", "Unknown")
                hops = record.get("hops", 1)
                related_entities.add(related_entity)
                
                # Score decreases with hop distance
                score = max(0.6, 0.8 - (hops * 0.1))
                
                chunks.append({
                    "chunk_id": chunk_id,
                    "document_id": record["document_id"],
                    "chunk_index": record["chunk_index"],
                    "chunk_type": record.get("chunk_type", "text"),
                    "chunk_text": record.get("chunk_text", ""),
                    "filename": record.get("filename", "Unknown Document"),
                    "graph_score": score,
                    "metadata": {
                        "section_title": record.get("section_title"),
                        "section_index": record.get("section_index"),
                        "related_entity": related_entity,
                        "hops": hops,
                        "retrieval_method": "entity_traversal",
                    }
                })
            
            query_duration = time.time() - query_start
            neo4j_graph_query_duration_seconds.labels(query_type="entity_traversal").observe(query_duration)
            neo4j_graph_queries_total.labels(query_type="entity_traversal").inc()
            neo4j_chunks_retrieved_via_graph.labels(query_type="entity_traversal").inc(len(chunks))
            
            if chunks:
                logger.info(
                    f"Entity traversal: found {len(chunks)} chunks via {len(related_entities)} related entities "
                    f"from starting entities: {entity_names[:3]}"
                )
            
            return chunks
        except Exception as e:
            logger.error(f"Entity traversal query failed: {str(e)}")
            return []
        finally:
            session.close()

