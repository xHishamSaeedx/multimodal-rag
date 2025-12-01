"""
Graph repository for Neo4j knowledge graph operations.

This module provides methods for building and managing the knowledge graph,
following best practices:
1. Store References, Not Duplicate Content
2. Batch Operations for Graph Building
3. Relationship Properties for Ranking/Weight
"""
from typing import List, Dict, Any, Optional
from neo4j import Driver
import time
from uuid import UUID

from app.core.neo4j_database import get_neo4j_driver
from app.core.config import settings
from app.core.database import DatabaseError
from app.repositories.graph_schema import (
    NODE_LABELS,
    RELATIONSHIP_TYPES,
    PROPERTY_KEYS,
    RELATIONSHIP_PROPERTIES,
)
from app.utils.metrics import (
    neo4j_query_duration_seconds,
    neo4j_queries_total,
    neo4j_nodes_created_total,
    neo4j_relationships_created_total,
    neo4j_graph_build_duration_seconds,
    neo4j_graph_builds_total,
    neo4j_transactions_total,
    neo4j_transaction_duration_seconds,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)


class GraphRepository:
    """
    Repository for Neo4j graph operations.
    
    Handles creating nodes, relationships, and querying the graph.
    """
    
    def __init__(self, driver: Optional[Driver] = None):
        """
        Initialize graph repository.
        
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
    
    def create_document_node(
        self,
        document_id: str,
        title: str,
        source: str,
        document_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Create a document node in the graph.
        
        Args:
            document_id: Unique document identifier (UUID as string)
            title: Document title
            source: Document source/path
            document_type: Type of document (optional)
            metadata: Lightweight metadata dict (optional)
        """
        query = f"""
        MERGE (d:{NODE_LABELS['DOCUMENT']} {{{PROPERTY_KEYS['DOCUMENT_ID']}: $document_id}})
        SET d.{PROPERTY_KEYS['TITLE']} = $title,
            d.{PROPERTY_KEYS['SOURCE']} = $source
        """
        
        params = {
            "document_id": document_id,
            "title": title,
            "source": source,
        }
        
        if document_type:
            query += f", d.{PROPERTY_KEYS['DOCUMENT_TYPE']} = $document_type"
            params["document_type"] = document_type
        
        if metadata:
            query += f", d.{PROPERTY_KEYS['METADATA']} = $metadata"
            params["metadata"] = metadata
        
        driver = self._get_driver()
        session = driver.session(database=self.database)
        try:
            session.run(query, params)
            logger.debug(f"Created document node: {document_id}")
        finally:
            session.close()
    
    def create_document_node(
        self,
        document_id: str,
        title: str,
        source: str,
        document_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Create just the document node (without sections/chunks).
        
        Useful when no chunks are available but we still want document metadata in graph.
        
        Args:
            document_id: Document UUID as string
            title: Document title
            source: Document source/path
            document_type: Type of document (optional)
            metadata: Lightweight metadata dict (optional)
        """
        query = f"""
        MERGE (d:{NODE_LABELS['DOCUMENT']} {{{PROPERTY_KEYS['DOCUMENT_ID']}: $document_id}})
        SET d.{PROPERTY_KEYS['TITLE']} = $title,
            d.{PROPERTY_KEYS['SOURCE']} = $source
        """
        
        params = {"document_id": document_id, "title": title, "source": source}
        
        if document_type:
            query += f", d.{PROPERTY_KEYS['DOCUMENT_TYPE']} = $document_type"
            params["document_type"] = document_type
        if metadata:
            query += f", d.{PROPERTY_KEYS['METADATA']} = $metadata"
            params["metadata"] = metadata
        
        driver = self._get_driver()
        session = driver.session(database=self.database)
        try:
            session.run(query, params)
            logger.info(f"Created document node: {document_id}")
        finally:
            session.close()
    
    def create_document_graph_batch(
        self,
        document_id: str,
        title: str,
        source: str,
        sections: List[Dict[str, Any]],
        document_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        skip_document_node: bool = False
    ) -> None:
        """
        Create entire document graph structure in a single batch transaction.
        
        This is 10-100x faster than creating nodes one-by-one.
        
        Args:
            document_id: Document UUID as string
            title: Document title
            source: Document source/path
            sections: List of section dictionaries, each containing:
                {
                    "section_id": str,
                    "title": str,
                    "index": int,
                    "chunks": [
                        {
                            "chunk_id": str (UUID as string),
                            "chunk_index": int,
                            "chunk_type": str,
                            "content": str (optional, for storing chunk text)
                            "metadata": dict (optional)
                        }
                    ]
                }
            document_type: Type of document (optional)
            metadata: Lightweight metadata dict (optional)
            skip_document_node: If True, skip creating/updating document node (useful for batch processing)
        """
        # Prepare document data structure
        document_data = {
            "document_id": document_id,
            "title": title,
            "source": source,
            "sections": sections,
        }
        
        if document_type:
            document_data["document_type"] = document_type
        if metadata:
            document_data["metadata"] = metadata
        
        if skip_document_node:
            # Only create sections and chunks, assuming document node already exists
            query = f"""
            MATCH (d:{NODE_LABELS['DOCUMENT']} {{{PROPERTY_KEYS['DOCUMENT_ID']}: $document_id}})
            UNWIND $sections AS section
            MERGE (s:{NODE_LABELS['SECTION']} {{{PROPERTY_KEYS['SECTION_ID']}: section.section_id}})
            SET s.{PROPERTY_KEYS['SECTION_TITLE']} = section.title,
                s.{PROPERTY_KEYS['SECTION_INDEX']} = section.index
            MERGE (d)-[r1:{RELATIONSHIP_TYPES['HAS_SECTION']} {{
                {RELATIONSHIP_PROPERTIES['ORDER']}: section.index
            }}]->(s)
            WITH s, section
            UNWIND section.chunks AS chunk
            WITH s, chunk
            WHERE chunk.chunk_id IS NOT NULL
            MERGE (c:{NODE_LABELS['CHUNK']} {{{PROPERTY_KEYS['CHUNK_ID']}: chunk.chunk_id}})
            SET c.{PROPERTY_KEYS['CHUNK_INDEX']} = COALESCE(chunk.chunk_index, 0),
                c.{PROPERTY_KEYS['CHUNK_TYPE']} = COALESCE(chunk.chunk_type, 'text'),
                c.{PROPERTY_KEYS['CONTENT']} = COALESCE(chunk.content, '')
            MERGE (s)-[r2:{RELATIONSHIP_TYPES['HAS_CHUNK']} {{
                {RELATIONSHIP_PROPERTIES['ORDER']}: COALESCE(chunk.chunk_index, 0),
                {RELATIONSHIP_PROPERTIES['CHUNK_INDEX']}: COALESCE(chunk.chunk_index, 0)
            }}]->(c)
            """
            query_params = {"document_id": document_id, "sections": sections}
            
            # Validate sections structure
            for section in sections:
                chunks_count = len(section.get("chunks", []))
                if chunks_count == 0:
                    logger.warning(f"Section {section.get('section_id')} has no chunks")
                else:
                    logger.debug(f"Section {section.get('section_id')} has {chunks_count} chunks")
        else:
            # Full query including document node creation
            query = f"""
            UNWIND $data AS doc
            MERGE (d:{NODE_LABELS['DOCUMENT']} {{{PROPERTY_KEYS['DOCUMENT_ID']}: doc.document_id}})
            SET d.{PROPERTY_KEYS['TITLE']} = doc.title,
                d.{PROPERTY_KEYS['SOURCE']} = doc.source
            """
            if document_type:
                query += f", d.{PROPERTY_KEYS['DOCUMENT_TYPE']} = doc.document_type"
            if metadata:
                query += f", d.{PROPERTY_KEYS['METADATA']} = doc.metadata"
            
            query += f"""
            WITH d, doc
            UNWIND doc.sections AS section
            MERGE (s:{NODE_LABELS['SECTION']} {{{PROPERTY_KEYS['SECTION_ID']}: section.section_id}})
            SET s.{PROPERTY_KEYS['SECTION_TITLE']} = section.title,
                s.{PROPERTY_KEYS['SECTION_INDEX']} = section.index
            MERGE (d)-[r1:{RELATIONSHIP_TYPES['HAS_SECTION']} {{
                {RELATIONSHIP_PROPERTIES['ORDER']}: section.index
            }}]->(s)
            WITH s, section
            UNWIND section.chunks AS chunk
            WITH s, chunk
            WHERE chunk.chunk_id IS NOT NULL
            MERGE (c:{NODE_LABELS['CHUNK']} {{{PROPERTY_KEYS['CHUNK_ID']}: chunk.chunk_id}})
            SET c.{PROPERTY_KEYS['CHUNK_INDEX']} = COALESCE(chunk.chunk_index, 0),
                c.{PROPERTY_KEYS['CHUNK_TYPE']} = COALESCE(chunk.chunk_type, 'text'),
                c.{PROPERTY_KEYS['CONTENT']} = COALESCE(chunk.content, '')
            MERGE (s)-[r2:{RELATIONSHIP_TYPES['HAS_CHUNK']} {{
                {RELATIONSHIP_PROPERTIES['ORDER']}: COALESCE(chunk.chunk_index, 0),
                {RELATIONSHIP_PROPERTIES['CHUNK_INDEX']}: COALESCE(chunk.chunk_index, 0)
            }}]->(c)
            """
            query_params = {"data": [document_data]}
            
            # Validate sections structure
            for section in sections:
                chunks_count = len(section.get("chunks", []))
                if chunks_count == 0:
                    logger.warning(f"Section {section.get('section_id')} has no chunks")
                else:
                    logger.debug(f"Section {section.get('section_id')} has {chunks_count} chunks")
        
        build_start = time.time()
        driver = self._get_driver()
        session = driver.session(database=self.database)
        try:
            # Log what we're about to create
            total_chunks = sum(len(s.get("chunks", [])) for s in sections)
            logger.info(
                f"Creating graph: {len(sections)} sections, {total_chunks} chunks, "
                f"skip_document_node={skip_document_node}"
            )
            
            # Validate and log section structure
            for idx, section in enumerate(sections):
                chunks_in_section = section.get("chunks", [])
                logger.info(
                    f"Section {idx + 1}/{len(sections)}: '{section.get('title', 'Unknown')}' "
                    f"with {len(chunks_in_section)} chunks"
                )
                if chunks_in_section:
                    # Log first chunk as sample
                    first_chunk = chunks_in_section[0]
                    logger.debug(
                        f"  Sample chunk: id={first_chunk.get('chunk_id')[:8]}..., "
                        f"type={first_chunk.get('chunk_type')}, "
                        f"content_length={len(first_chunk.get('content', ''))}"
                    )
            
            with session.begin_transaction() as tx:
                result = tx.run(query, query_params)
                # Consume result to get execution info
                summary = result.consume()
                logger.debug(
                    f"Neo4j query executed: nodes_created={summary.counters.nodes_created}, "
                    f"relationships_created={summary.counters.relationships_created}"
                )
                tx.commit()
            
            # Record metrics
            build_duration = time.time() - build_start
            neo4j_graph_build_duration_seconds.labels(document_type=document_type or "unknown").observe(build_duration)
            neo4j_graph_builds_total.labels(status="success").inc()
            
            # Count nodes and relationships created
            total_nodes = 1  # Document node
            total_relationships = 0
            for section in sections:
                total_nodes += 1  # Section node
                total_relationships += 1  # HAS_SECTION
                for chunk in section.get("chunks", []):
                    total_nodes += 1  # Chunk node
                    total_relationships += 1  # HAS_CHUNK
            
            neo4j_nodes_created_total.labels(node_type="Document").inc()
            neo4j_nodes_created_total.labels(node_type="Section").inc(len(sections))
            neo4j_nodes_created_total.labels(node_type="Chunk").inc(sum(len(s.get("chunks", [])) for s in sections))
            neo4j_relationships_created_total.labels(relationship_type="HAS_SECTION").inc(len(sections))
            neo4j_relationships_created_total.labels(relationship_type="HAS_CHUNK").inc(sum(len(s.get("chunks", [])) for s in sections))
            
            logger.info(f"Created document graph for: {document_id}")
        except Exception as e:
            neo4j_graph_builds_total.labels(status="error").inc()
            neo4j_queries_total.labels(operation="create_document_graph", status="error").inc()
            logger.error(f"Failed to create document graph: {str(e)}")
            raise
        finally:
            session.close()
    
    def create_entity_nodes_batch(
        self,
        entities: List[Dict[str, Any]]
    ) -> None:
        """
        Create multiple entity nodes in a single batch operation.
        
        Uses normalized_name (lowercase, trimmed) for cross-document entity resolution.
        Entities with the same normalized_name will share a single node, enabling
        cross-document discovery.
        
        Args:
            entities: List of entity dictionaries, each containing:
                {
                    "entity_id": str,
                    "entity_name": str,
                    "entity_type": str,
                    "normalized_name": str (optional, computed if not provided),
                    "entity_value": str (optional, for metrics),
                    "confidence": float (optional)
                }
        """
        if not entities:
            return
        
        # Normalize entity names for cross-document matching
        for entity in entities:
            if 'normalized_name' not in entity or not entity['normalized_name']:
                entity['normalized_name'] = entity['entity_name'].lower().strip()
        
        # Use normalized_name for MERGE to enable cross-document entity resolution
        # Same entity across documents will share one node
        query = f"""
        UNWIND $entities AS entity
        MERGE (e:{NODE_LABELS['ENTITY']} {{{PROPERTY_KEYS['ENTITY_NORMALIZED_NAME']}: entity.normalized_name}})
        ON CREATE SET e.{PROPERTY_KEYS['ENTITY_ID']} = entity.entity_id
        SET e.{PROPERTY_KEYS['ENTITY_NAME']} = entity.entity_name,
            e.{PROPERTY_KEYS['ENTITY_TYPE']} = entity.entity_type,
            e.{PROPERTY_KEYS['ENTITY_NORMALIZED_NAME']} = entity.normalized_name
        """
        
        # Add optional fields if present
        if entities and 'entity_value' in entities[0] and entities[0].get('entity_value'):
            query += f", e.{PROPERTY_KEYS['ENTITY_VALUE']} = entity.entity_value"
        
        if entities and 'confidence' in entities[0] and entities[0].get('confidence') is not None:
            query += f", e.{PROPERTY_KEYS['CONFIDENCE']} = entity.confidence"
        
        query_start = time.time()
        driver = self._get_driver()
        session = driver.session(database=self.database)
        try:
            with session.begin_transaction() as tx:
                tx.run(query, entities=entities)
                tx.commit()
            
            # Record metrics
            query_duration = time.time() - query_start
            neo4j_query_duration_seconds.labels(operation="create_entities").observe(query_duration)
            neo4j_queries_total.labels(operation="create_entities", status="success").inc()
            neo4j_nodes_created_total.labels(node_type="Entity").inc(len(entities))
            neo4j_transactions_total.labels(status="success").inc()
            neo4j_transaction_duration_seconds.observe(query_duration)
            
            logger.info(f"Created/merged {len(entities)} entity nodes in batch")
        except Exception as e:
            query_duration = time.time() - query_start
            neo4j_queries_total.labels(operation="create_entities", status="error").inc()
            neo4j_transactions_total.labels(status="error").inc()
            neo4j_transaction_duration_seconds.observe(query_duration)
            logger.error(f"Failed to create entity nodes: {str(e)}")
            raise
        finally:
            session.close()
    
    def create_relationships_batch(
        self,
        relationships: List[Dict[str, Any]]
    ) -> None:
        """
        Create multiple relationships in a single batch operation.
        
        Args:
            relationships: List of relationship dictionaries with:
                {
                    "from_label": str,
                    "from_id_key": str,
                    "from_id_value": str,
                    "to_label": str,
                    "to_id_key": str,
                    "to_id_value": str,
                    "relationship_type": str,
                    "properties": dict (optional)
                }
        """
        if not relationships:
            return
        
        query_start = time.time()
        driver = self._get_driver()
        session = driver.session(database=self.database)
        try:
            with session.begin_transaction() as tx:
                for rel in relationships:
                    from_label = rel['from_label']
                    from_id_key = rel['from_id_key']
                    from_id_value = rel['from_id_value']
                    to_label = rel['to_label']
                    to_id_key = rel['to_id_key']
                    to_id_value = rel['to_id_value']
                    rel_type = rel['relationship_type']
                    properties = rel.get('properties', {})
                    
                    # Build query with relationship type
                    if properties:
                        props_str = ", ".join([f"{k}: ${k}" for k in properties.keys()])
                        query = f"""
                        MATCH (from:{from_label} {{{from_id_key}: $from_id}})
                        MATCH (to:{to_label} {{{to_id_key}: $to_id}})
                        MERGE (from)-[r:{rel_type}]->(to)
                        SET r += {{{props_str}}}
                        """
                        params = {
                            "from_id": from_id_value,
                            "to_id": to_id_value,
                            **properties
                        }
                    else:
                        query = f"""
                        MATCH (from:{from_label} {{{from_id_key}: $from_id}})
                        MATCH (to:{to_label} {{{to_id_key}: $to_id}})
                        MERGE (from)-[r:{rel_type}]->(to)
                        """
                        params = {
                            "from_id": from_id_value,
                            "to_id": to_id_value,
                        }
                    
                    tx.run(query, params)
                
                tx.commit()
            
            # Record metrics
            query_duration = time.time() - query_start
            neo4j_query_duration_seconds.labels(operation="create_relationships").observe(query_duration)
            neo4j_queries_total.labels(operation="create_relationships", status="success").inc()
            
            # Count relationships by type
            rel_type_counts = {}
            for rel in relationships:
                rel_type = rel['relationship_type']
                rel_type_counts[rel_type] = rel_type_counts.get(rel_type, 0) + 1
            
            for rel_type, count in rel_type_counts.items():
                neo4j_relationships_created_total.labels(relationship_type=rel_type).inc(count)
            
            neo4j_transactions_total.labels(status="success").inc()
            neo4j_transaction_duration_seconds.observe(query_duration)
            
            logger.info(f"Created {len(relationships)} relationships in batch")
        except Exception as e:
            query_duration = time.time() - query_start
            neo4j_queries_total.labels(operation="create_relationships", status="error").inc()
            neo4j_transactions_total.labels(status="error").inc()
            neo4j_transaction_duration_seconds.observe(query_duration)
            logger.error(f"Failed to create relationships: {str(e)}")
            raise
        finally:
            session.close()
    
    def create_mentions_relationship(
        self,
        chunk_id: str,
        entity_id: str,
        frequency: int = 1,
        importance: float = 0.5,
        context: Optional[str] = None
    ) -> None:
        """
        Create MENTIONS relationship with importance/weight properties.
        
        Args:
            chunk_id: Chunk identifier (UUID as string)
            entity_id: Entity identifier
            frequency: How often entity appears in chunk
            importance: Relevance score (0.0-1.0)
            context: Where entity appears (title, body, metadata)
        """
        properties = {
            RELATIONSHIP_PROPERTIES['FREQUENCY']: frequency,
            RELATIONSHIP_PROPERTIES['IMPORTANCE']: importance,
        }
        
        if context:
            properties[RELATIONSHIP_PROPERTIES['CONTEXT']] = context
        
        self.create_relationships_batch([{
            "from_label": NODE_LABELS['CHUNK'],
            "from_id_key": PROPERTY_KEYS['CHUNK_ID'],
            "from_id_value": chunk_id,
            "to_label": NODE_LABELS['ENTITY'],
            "to_id_key": PROPERTY_KEYS['ENTITY_ID'],
            "to_id_value": entity_id,
            "relationship_type": RELATIONSHIP_TYPES['MENTIONS'],
            "properties": properties
        }])
    
    def create_about_relationship(
        self,
        section_id: str,
        entity_id: str,
        importance: float = 0.5,
        frequency: int = 1
    ) -> None:
        """
        Create ABOUT relationship with importance/weight properties.
        
        Args:
            section_id: Section identifier
            entity_id: Entity identifier
            importance: Relevance score (0.0-1.0)
            frequency: How often entity appears in section
        """
        properties = {
            RELATIONSHIP_PROPERTIES['IMPORTANCE']: importance,
            RELATIONSHIP_PROPERTIES['FREQUENCY']: frequency,
        }
        
        self.create_relationships_batch([{
            "from_label": NODE_LABELS['SECTION'],
            "from_id_key": PROPERTY_KEYS['SECTION_ID'],
            "from_id_value": section_id,
            "to_label": NODE_LABELS['ENTITY'],
            "to_id_key": PROPERTY_KEYS['ENTITY_ID'],
            "to_id_value": entity_id,
            "relationship_type": RELATIONSHIP_TYPES['ABOUT'],
            "properties": properties
        }])
    
    def create_next_chunk_relationships_batch(
        self,
        chunk_pairs: List[Dict[str, str]]
    ) -> None:
        """
        Create NEXT_CHUNK relationships between consecutive chunks.
        
        This enables context expansion during retrieval - given a chunk,
        we can easily find its neighbors for more context.
        
        Args:
            chunk_pairs: List of dictionaries with:
                {
                    "from_chunk_id": str,  # Current chunk
                    "to_chunk_id": str,    # Next chunk
                }
        """
        if not chunk_pairs:
            return
        
        query = f"""
        UNWIND $pairs AS pair
        MATCH (c1:{NODE_LABELS['CHUNK']} {{{PROPERTY_KEYS['CHUNK_ID']}: pair.from_chunk_id}})
        MATCH (c2:{NODE_LABELS['CHUNK']} {{{PROPERTY_KEYS['CHUNK_ID']}: pair.to_chunk_id}})
        MERGE (c1)-[r:{RELATIONSHIP_TYPES['NEXT_CHUNK']}]->(c2)
        """
        
        query_start = time.time()
        driver = self._get_driver()
        session = driver.session(database=self.database)
        try:
            with session.begin_transaction() as tx:
                tx.run(query, pairs=chunk_pairs)
                tx.commit()
            
            query_duration = time.time() - query_start
            neo4j_query_duration_seconds.labels(operation="create_next_chunk").observe(query_duration)
            neo4j_queries_total.labels(operation="create_next_chunk", status="success").inc()
            neo4j_relationships_created_total.labels(relationship_type="NEXT_CHUNK").inc(len(chunk_pairs))
            
            logger.info(f"Created {len(chunk_pairs)} NEXT_CHUNK relationships")
        except Exception as e:
            query_duration = time.time() - query_start
            neo4j_queries_total.labels(operation="create_next_chunk", status="error").inc()
            logger.error(f"Failed to create NEXT_CHUNK relationships: {str(e)}")
            raise
        finally:
            session.close()
    
    def create_topic_nodes_batch(
        self,
        topics: List[Dict[str, Any]]
    ) -> None:
        """
        Create multiple topic nodes in a single batch operation.
        
        Topics serve as a global ontology layer for cross-document navigation.
        Same topics across documents will share one node via normalized_name.
        
        Args:
            topics: List of topic dictionaries, each containing:
                {
                    "topic_id": str,
                    "topic_name": str,
                    "normalized_name": str (optional, computed if not provided),
                    "keywords": List[str] (optional),
                    "description": str (optional)
                }
        """
        if not topics:
            return
        
        # Normalize topic names for cross-document matching
        for topic in topics:
            if 'normalized_name' not in topic or not topic['normalized_name']:
                topic['normalized_name'] = topic['topic_name'].lower().strip()
        
        # Use normalized_name for MERGE to enable cross-document topic resolution
        query = f"""
        UNWIND $topics AS topic
        MERGE (t:{NODE_LABELS['TOPIC']} {{{PROPERTY_KEYS['TOPIC_NORMALIZED_NAME']}: topic.normalized_name}})
        ON CREATE SET t.{PROPERTY_KEYS['TOPIC_ID']} = topic.topic_id
        SET t.{PROPERTY_KEYS['TOPIC_NAME']} = topic.topic_name,
            t.{PROPERTY_KEYS['TOPIC_NORMALIZED_NAME']} = topic.normalized_name
        """
        
        # Add optional fields if present
        if topics and 'keywords' in topics[0]:
            query += f", t.{PROPERTY_KEYS['TOPIC_KEYWORDS']} = topic.keywords"
        
        if topics and 'description' in topics[0]:
            query += f", t.{PROPERTY_KEYS['TOPIC_DESCRIPTION']} = topic.description"
        
        query_start = time.time()
        driver = self._get_driver()
        session = driver.session(database=self.database)
        try:
            with session.begin_transaction() as tx:
                tx.run(query, topics=topics)
                tx.commit()
            
            query_duration = time.time() - query_start
            neo4j_query_duration_seconds.labels(operation="create_topics").observe(query_duration)
            neo4j_queries_total.labels(operation="create_topics", status="success").inc()
            neo4j_nodes_created_total.labels(node_type="Topic").inc(len(topics))
            
            logger.info(f"Created/merged {len(topics)} topic nodes in batch")
        except Exception as e:
            query_duration = time.time() - query_start
            neo4j_queries_total.labels(operation="create_topics", status="error").inc()
            logger.error(f"Failed to create topic nodes: {str(e)}")
            raise
        finally:
            session.close()
    
    def create_entity_relationships_batch(
        self,
        entity_pairs: List[Dict[str, Any]]
    ) -> None:
        """
        Create RELATED_TO relationships between entities.
        
        These represent semantic relationships extracted from text co-occurrence,
        or explicitly extracted relationships (e.g., "works for", "located in").
        
        Args:
            entity_pairs: List of dictionaries with:
                {
                    "from_entity_normalized_name": str,
                    "to_entity_normalized_name": str,
                    "relationship_type": str (optional, defaults to "RELATED_TO"),
                    "context": str (optional),
                    "confidence": float (optional)
                }
        """
        if not entity_pairs:
            return
        
        query = f"""
        UNWIND $pairs AS pair
        MATCH (e1:{NODE_LABELS['ENTITY']} {{{PROPERTY_KEYS['ENTITY_NORMALIZED_NAME']}: pair.from_entity}})
        MATCH (e2:{NODE_LABELS['ENTITY']} {{{PROPERTY_KEYS['ENTITY_NORMALIZED_NAME']}: pair.to_entity}})
        MERGE (e1)-[r:{RELATIONSHIP_TYPES['RELATED_TO']}]->(e2)
        SET r.context = COALESCE(pair.context, ''),
            r.confidence = COALESCE(pair.confidence, 0.5)
        """
        
        # Prepare pairs data
        prepared_pairs = []
        for pair in entity_pairs:
            prepared_pairs.append({
                "from_entity": pair["from_entity_normalized_name"],
                "to_entity": pair["to_entity_normalized_name"],
                "context": pair.get("context", ""),
                "confidence": pair.get("confidence", 0.5)
            })
        
        query_start = time.time()
        driver = self._get_driver()
        session = driver.session(database=self.database)
        try:
            with session.begin_transaction() as tx:
                tx.run(query, pairs=prepared_pairs)
                tx.commit()
            
            query_duration = time.time() - query_start
            neo4j_query_duration_seconds.labels(operation="create_entity_relations").observe(query_duration)
            neo4j_queries_total.labels(operation="create_entity_relations", status="success").inc()
            neo4j_relationships_created_total.labels(relationship_type="RELATED_TO").inc(len(entity_pairs))
            
            logger.info(f"Created {len(entity_pairs)} entity-to-entity relationships")
        except Exception as e:
            query_duration = time.time() - query_start
            neo4j_queries_total.labels(operation="create_entity_relations", status="error").inc()
            logger.error(f"Failed to create entity relationships: {str(e)}")
            raise
        finally:
            session.close()
    
    def delete_document_graph(self, document_id: str) -> Dict[str, int]:
        """
        Safely delete a document and all its related nodes from the knowledge graph.
        
        Deletion Strategy:
        1. Identify entities that will become orphaned (only mentioned in this doc)
        2. Identify media that will become orphaned (only attached to this doc)
        3. Delete all Chunk nodes for this document
        4. Delete orphaned Media nodes
        5. Delete orphaned Entity nodes
        6. Delete all Section nodes
        7. Delete orphaned Topic nodes (topics only linked to this doc)
        8. Delete the Document node itself
        
        This ensures:
        - Complete cleanup of document-specific content
        - Preservation of cross-document entities (shared entities)
        - Cleanup of topics when last document is deleted
        
        Args:
            document_id: Document identifier (UUID as string)
        
        Returns:
            Dictionary with deletion counts
        """
        driver = self._get_driver()
        session = driver.session(database=self.database)
        
        try:
            # Step 1: Identify entities that will become orphaned BEFORE deleting chunks
            # (entities that are ONLY mentioned in this document's chunks)
            orphan_entities_query = f"""
            MATCH (d:{NODE_LABELS['DOCUMENT']} {{{PROPERTY_KEYS['DOCUMENT_ID']}: $document_id}})
                  -[:{RELATIONSHIP_TYPES['HAS_SECTION']}]->(:{NODE_LABELS['SECTION']})
                  -[:{RELATIONSHIP_TYPES['HAS_CHUNK']}]->(c:{NODE_LABELS['CHUNK']})
                  -[:{RELATIONSHIP_TYPES['MENTIONS']}]->(e:{NODE_LABELS['ENTITY']})
            WITH DISTINCT e
            // Check if entity is mentioned in any OTHER chunks (outside this document)
            WHERE NOT EXISTS {{
                MATCH (other_chunk:{NODE_LABELS['CHUNK']})-[:{RELATIONSHIP_TYPES['MENTIONS']}]->(e)
                WHERE NOT EXISTS {{
                    MATCH (d:{NODE_LABELS['DOCUMENT']} {{{PROPERTY_KEYS['DOCUMENT_ID']}: $document_id}})
                          -[:{RELATIONSHIP_TYPES['HAS_SECTION']}]->(:{NODE_LABELS['SECTION']})
                          -[:{RELATIONSHIP_TYPES['HAS_CHUNK']}]->(other_chunk)
                }}
            }}
            RETURN collect(e.{PROPERTY_KEYS['ENTITY_ID']}) AS orphan_entity_ids, count(e) AS orphan_count
            """
            result = session.run(orphan_entities_query, document_id=document_id)
            record = result.single()
            orphan_entity_ids = record["orphan_entity_ids"] if record else []
            orphan_entities_count = record["orphan_count"] if record else 0
            logger.debug(f"Identified {orphan_entities_count} entities that will become orphaned")
            
            # Step 2: Identify media that will become orphaned BEFORE deleting chunks
            orphan_media_query = f"""
            MATCH (d:{NODE_LABELS['DOCUMENT']} {{{PROPERTY_KEYS['DOCUMENT_ID']}: $document_id}})
                  -[:{RELATIONSHIP_TYPES['HAS_SECTION']}]->(:{NODE_LABELS['SECTION']})
                  -[:{RELATIONSHIP_TYPES['HAS_CHUNK']}]->(c:{NODE_LABELS['CHUNK']})
                  -[:{RELATIONSHIP_TYPES['HAS_MEDIA']}]->(m:{NODE_LABELS['MEDIA']})
            WITH DISTINCT m
            // Check if media is attached to any OTHER chunks
            WHERE NOT EXISTS {{
                MATCH (other_chunk:{NODE_LABELS['CHUNK']})-[:{RELATIONSHIP_TYPES['HAS_MEDIA']}]->(m)
                WHERE NOT EXISTS {{
                    MATCH (d:{NODE_LABELS['DOCUMENT']} {{{PROPERTY_KEYS['DOCUMENT_ID']}: $document_id}})
                          -[:{RELATIONSHIP_TYPES['HAS_SECTION']}]->(:{NODE_LABELS['SECTION']})
                          -[:{RELATIONSHIP_TYPES['HAS_CHUNK']}]->(other_chunk)
                }}
            }}
            RETURN collect(m.{PROPERTY_KEYS['MEDIA_ID']}) AS orphan_media_ids, count(m) AS orphan_count
            """
            result = session.run(orphan_media_query, document_id=document_id)
            record = result.single()
            orphan_media_ids = record["orphan_media_ids"] if record else []
            orphan_media_count = record["orphan_count"] if record else 0
            logger.debug(f"Identified {orphan_media_count} media nodes that will become orphaned")
            
            # Step 3: Delete all chunks for this document
            # This removes MENTIONS, HAS_TOPIC, HAS_MEDIA, NEXT_CHUNK relationships
            chunks_query = f"""
            MATCH (d:{NODE_LABELS['DOCUMENT']} {{{PROPERTY_KEYS['DOCUMENT_ID']}: $document_id}})
                  -[:{RELATIONSHIP_TYPES['HAS_SECTION']}]->(s:{NODE_LABELS['SECTION']})
                  -[:{RELATIONSHIP_TYPES['HAS_CHUNK']}]->(c:{NODE_LABELS['CHUNK']})
            WITH collect(c) AS chunks
            UNWIND chunks AS chunk
            DETACH DELETE chunk
            RETURN count(chunk) AS chunks_deleted
            """
            result = session.run(chunks_query, document_id=document_id)
            record = result.single()
            chunks_deleted = record["chunks_deleted"] if record else 0
            logger.info(f"Deleted {chunks_deleted} chunks for document {document_id}")
            
            # Step 4: Delete the identified orphaned entities
            entities_deleted = 0
            if orphan_entity_ids:
                delete_entities_query = f"""
                MATCH (e:{NODE_LABELS['ENTITY']})
                WHERE e.{PROPERTY_KEYS['ENTITY_ID']} IN $entity_ids
                DETACH DELETE e
                RETURN count(e) AS deleted_count
                """
                result = session.run(delete_entities_query, entity_ids=orphan_entity_ids)
                record = result.single()
                entities_deleted = record["deleted_count"] if record else 0
                if entities_deleted > 0:
                    logger.info(f"Deleted {entities_deleted} orphaned entity nodes")
            
            # Step 4b: Delete completely disconnected entities (no relationships at all)
            # This handles entities that were created but never linked due to bugs
            disconnected_entities_query = f"""
            MATCH (e:{NODE_LABELS['ENTITY']})
            WHERE NOT (e)--()
            DETACH DELETE e
            RETURN count(e) AS deleted_count
            """
            result = session.run(disconnected_entities_query)
            record = result.single()
            disconnected_entities_deleted = record["deleted_count"] if record else 0
            if disconnected_entities_deleted > 0:
                logger.info(f"Deleted {disconnected_entities_deleted} completely disconnected entity nodes")
                entities_deleted += disconnected_entities_deleted
            
            # Step 4c: Delete entities that have no connection to any chunks
            # (they might be connected to other entities via RELATED_TO but are not mentioned in any chunk)
            unlinked_entities_query = f"""
            MATCH (e:{NODE_LABELS['ENTITY']})
            WHERE NOT (e)<-[:{RELATIONSHIP_TYPES['MENTIONS']}]-()
            AND NOT (e)<-[:{RELATIONSHIP_TYPES['ABOUT']}]-()
            DETACH DELETE e
            RETURN count(e) AS deleted_count
            """
            result = session.run(unlinked_entities_query)
            record = result.single()
            unlinked_entities_deleted = record["deleted_count"] if record else 0
            if unlinked_entities_deleted > 0:
                logger.info(f"Deleted {unlinked_entities_deleted} entities with no chunk connections")
                entities_deleted += unlinked_entities_deleted
            
            # Step 5: Delete the identified orphaned media
            media_deleted = 0
            if orphan_media_ids:
                delete_media_query = f"""
                MATCH (m:{NODE_LABELS['MEDIA']})
                WHERE m.{PROPERTY_KEYS['MEDIA_ID']} IN $media_ids
                DETACH DELETE m
                RETURN count(m) AS deleted_count
                """
                result = session.run(delete_media_query, media_ids=orphan_media_ids)
                record = result.single()
                media_deleted = record["deleted_count"] if record else 0
                if media_deleted > 0:
                    logger.info(f"Deleted {media_deleted} orphaned media nodes")
            
            # Step 6: Delete all sections for this document
            sections_query = f"""
            MATCH (d:{NODE_LABELS['DOCUMENT']} {{{PROPERTY_KEYS['DOCUMENT_ID']}: $document_id}})
                  -[:{RELATIONSHIP_TYPES['HAS_SECTION']}]->(s:{NODE_LABELS['SECTION']})
            DETACH DELETE s
            RETURN count(s) AS sections_deleted
            """
            result = session.run(sections_query, document_id=document_id)
            record = result.single()
            sections_deleted = record["sections_deleted"] if record else 0
            logger.info(f"Deleted {sections_deleted} sections for document {document_id}")
            
            # Step 7: Delete orphaned Topic nodes (topics with no remaining chunk/section links)
            # Topics are meant to be global, but if no documents use them anymore, clean them up
            topics_query = f"""
            MATCH (t:{NODE_LABELS['TOPIC']})
            WHERE NOT EXISTS {{
                MATCH (:{NODE_LABELS['CHUNK']})-[:{RELATIONSHIP_TYPES['HAS_TOPIC']}]->(t)
            }} AND NOT EXISTS {{
                MATCH (:{NODE_LABELS['SECTION']})-[:{RELATIONSHIP_TYPES['HAS_TOPIC']}]->(t)
            }}
            DETACH DELETE t
            RETURN count(t) AS topics_deleted
            """
            result = session.run(topics_query)
            record = result.single()
            topics_deleted = record["topics_deleted"] if record else 0
            if topics_deleted > 0:
                logger.info(f"Deleted {topics_deleted} orphaned topic nodes")
            
            # Step 8: Delete the document node itself
            doc_query = f"""
            MATCH (d:{NODE_LABELS['DOCUMENT']} {{{PROPERTY_KEYS['DOCUMENT_ID']}: $document_id}})
            DETACH DELETE d
            RETURN count(d) AS doc_deleted
            """
            result = session.run(doc_query, document_id=document_id)
            record = result.single()
            doc_deleted = record["doc_deleted"] if record else 0
            
            counts = {
                "document_deleted": doc_deleted,
                "sections_deleted": sections_deleted,
                "chunks_deleted": chunks_deleted,
                "media_deleted": media_deleted,
                "orphaned_entities_deleted": entities_deleted,
                "orphaned_topics_deleted": topics_deleted,
            }
            
            logger.info(
                f"Successfully deleted document graph: {document_id}",
                extra={"deletion_counts": counts}
            )
            
            return counts
            
        except Exception as e:
            logger.error(
                f"Failed to delete document graph: {document_id}",
                extra={"error": str(e)},
                exc_info=True
            )
            raise DatabaseError(f"Graph deletion failed: {str(e)}")
        finally:
            session.close()
    
    def cleanup_orphaned_entities(self) -> Dict[str, int]:
        """
        Clean up orphaned entities across the entire database.
        
        Removes:
        1. Entities with no relationships at all (completely disconnected)
        2. Entities not connected to any chunks (no MENTIONS or ABOUT relationships)
        
        This is useful for cleaning up after bugs where entities were created
        but never properly linked to document chunks.
        
        Returns:
            Dictionary with deletion counts
        """
        driver = self._get_driver()
        session = driver.session(database=self.database)
        
        try:
            # Step 1: Delete completely disconnected entities
            disconnected_query = f"""
            MATCH (e:{NODE_LABELS['ENTITY']})
            WHERE NOT (e)--()
            DETACH DELETE e
            RETURN count(e) AS deleted_count
            """
            result = session.run(disconnected_query)
            record = result.single()
            disconnected_deleted = record["deleted_count"] if record else 0
            
            # Step 2: Delete entities with no chunk connections
            unlinked_query = f"""
            MATCH (e:{NODE_LABELS['ENTITY']})
            WHERE NOT (e)<-[:{RELATIONSHIP_TYPES['MENTIONS']}]-()
            AND NOT (e)<-[:{RELATIONSHIP_TYPES['ABOUT']}]-()
            DETACH DELETE e
            RETURN count(e) AS deleted_count
            """
            result = session.run(unlinked_query)
            record = result.single()
            unlinked_deleted = record["deleted_count"] if record else 0
            
            counts = {
                "disconnected_entities_deleted": disconnected_deleted,
                "unlinked_entities_deleted": unlinked_deleted,
                "total_entities_deleted": disconnected_deleted + unlinked_deleted,
            }
            
            logger.info(f"Cleaned up orphaned entities: {counts}")
            return counts
        finally:
            session.close()
    
    def delete_all_graph_data(self) -> Dict[str, int]:
        """
        Delete ALL nodes and relationships from Neo4j (complete cleanup).
        
        WARNING: This will delete everything in the graph!
        
        Returns:
            Dictionary with deletion counts:
            {
                "nodes_deleted": int,
                "relationships_deleted": int
            }
        """
        query = "MATCH (n) DETACH DELETE n"
        
        driver = self._get_driver()
        session = driver.session(database=self.database)
        try:
            result = session.run(query)
            summary = result.consume()
            
            counts = {
                "nodes_deleted": summary.counters.nodes_deleted,
                "relationships_deleted": summary.counters.relationships_deleted,
            }
            
            logger.warning(f"Deleted ALL graph data: {counts}")
            return counts
        finally:
            session.close()

