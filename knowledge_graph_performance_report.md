# Knowledge Graph Performance Report

## Overview

This report analyzes the knowledge graph retrieval performance based on 5 queries executed in the past hour. The analysis covers both unified knowledge graph retrieval times and detailed performance by query type.

## Data Collection

- **Time Period**: Past 1 hour
- **Total Queries**: 5
- **Data Source**: Prometheus metrics via Grafana dashboard
- **Metrics Collected**: Retrieval times, query durations by type, chunk retrieval counts

## Unified Knowledge Graph Retrieval Performance

### Average Retrieval Speed

- **Knowledge Graph Unified Average**: 243 ms
- **Knowledge Graph Max Retrieval Time**: 507 ms
- **Total Knowledge Graph Queries Executed**: 20 (4 query types × 5 queries each)

## Knowledge Graph Query Duration by Type

The following table shows the average performance for each knowledge graph query type:

| Query Type         | Average Duration | Max Duration | Queries Executed |
| ------------------ | ---------------- | ------------ | ---------------- |
| `by_keywords`      | 670 ms           | 1.21 s       | 5                |
| `by_section_title` | 223 ms           | 422 ms       | 5                |
| `by_topics`        | 119 ms           | 214 ms       | 5                |
| `graph_traversal`  | 107 ms           | 186 ms       | 5                |

## Performance Analysis

### Query Type Performance Ranking (Fastest to Slowest)

1. **graph_traversal**: 107 ms average (fastest)
2. **by_topics**: 119 ms average
3. **by_section_title**: 223 ms average
4. **by_keywords**: 670 ms average (slowest)

### Key Observations

- **graph_traversal** is the most efficient query type with the lowest average latency (107 ms)
- **by_keywords** queries take significantly longer (670 ms average), likely due to more complex matching requirements
- All query types show consistent performance with reasonable maximum latencies
- The unified knowledge graph retrieval time (243 ms) represents the combined performance across all query types

### Chunk Retrieval Performance

- **Total Chunks Retrieved via Graph**: 94 chunks
- **Average Chunks per Query**: 18.7 chunks
- **Graph vs Hybrid Retrieval**: Graph retrieval provides ~87% more chunks per query than hybrid retrieval (18.7 vs 10.0)

## Recommendations

1. **Optimize by_keywords queries**: The 670ms average suggests potential for optimization in keyword-based searches
2. **Leverage graph_traversal**: This appears to be the most efficient retrieval method
3. **Consider query type selection**: For performance-critical applications, prefer graph_traversal or by_topics over by_keywords
4. **Monitor max latencies**: While averages are good, the 1.21s max for by_keywords should be monitored for potential outliers

## Methodology

- Metrics collected from Prometheus using the following queries:
  - `retrieval_duration_seconds_sum{retrieval_type="graph"} / retrieval_duration_seconds_count{retrieval_type="graph"}` for unified average
  - `sum(neo4j_graph_query_duration_seconds_sum) by (query_type) / sum(neo4j_graph_query_duration_seconds_count) by (query_type)` for type-specific averages
- All averages calculated over the 5 queries executed in the past hour
- Time range: Last 1 hour from report generation

## Conclusion

The knowledge graph retrieval system demonstrates solid performance with sub-second average retrieval times. The system successfully handles multiple query types with varying complexity, with graph_traversal being the most efficient method. The unified 243ms average retrieval time indicates good overall system performance for knowledge graph-based retrieval operations.
