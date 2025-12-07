# Scripts

Utility scripts for managing the multimodal-rag system.

## cleanup_neo4j.py

Cleans out the entire Neo4j database - deletes all nodes and relationships.

**Usage:**
```bash
# Interactive (with confirmation prompt)
cd backend
python scripts/neo4j/cleanup_neo4j.py

# Non-interactive (skip confirmation)
python scripts/neo4j/cleanup_neo4j.py --confirm
```

**What it does:**
- Connects to Neo4j using settings from `app/core/config.py`
- Deletes ALL nodes and relationships using `MATCH (n) DETACH DELETE n`
- Reports how many nodes and relationships were deleted

**Warning:** This will delete everything! Use with caution.

## Notes

- Scripts should be run from the `backend/` directory
- Make sure your `.env` file is configured with correct Neo4j credentials
- The cleanup script uses the same Neo4j connection settings as the main application

