import React from "react";
import "./About.css";

const About: React.FC = () => {
  return (
    <section id="about" className="about">
      <div className="about-container">
        <div className="about-header">
          <h2 className="about-title">Complete ML Research Implementation</h2>
          <p className="about-subtitle">
            Full-stack ML demonstration showcasing advanced retrieval
            strategies, multimodal processing, and comprehensive observability
            for research and educational purposes
          </p>
        </div>

        <div className="about-grid">
          <div className="about-card">
            <svg
              className="card-icon"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <line x1="12" y1="20" x2="12" y2="10"></line>
              <line x1="18" y1="20" x2="18" y2="4"></line>
              <line x1="6" y1="20" x2="6" y2="16"></line>
            </svg>
            <h3 className="card-title">Ensemble Retrieval Implementation</h3>
            <p className="card-description">
              Demonstrates ensemble retrieval combining BM25 sparse search and
              dense vector similarity with fusion techniques. Shows
              metadata-aware reranking and multi-stage filtering for improved
              retrieval precision across heterogeneous data sources.
            </p>
          </div>

          <div className="about-card">
            <svg
              className="card-icon"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <rect x="3" y="3" width="7" height="7"></rect>
              <rect x="14" y="3" width="7" height="7"></rect>
              <rect x="14" y="14" width="7" height="7"></rect>
              <rect x="3" y="14" width="7" height="7"></rect>
            </svg>
            <h3 className="card-title">Cross-Modal Feature Pipeline</h3>
            <p className="card-description">
              Demonstrates cross-modal feature extraction processing text,
              images, and tabular data. Implements vision-language models for
              image understanding, automated document layout analysis, and
              hierarchical chunking strategies for optimal embedding quality.
            </p>
          </div>

          <div className="about-card">
            <svg
              className="card-icon"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"></path>
              <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"></path>
            </svg>
            <h3 className="card-title">Graph-Based Reasoning Demo</h3>
            <p className="card-description">
              Demonstrates graph-based reasoning layer with Neo4j for complex
              relationship modeling and multi-hop inference. Shows advanced RAG
              patterns combining vector similarity with structured knowledge
              representations for enhanced contextual understanding.
            </p>
          </div>

          <div className="about-card">
            <svg
              className="card-icon"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <circle cx="12" cy="12" r="3"></circle>
              <path d="M12 1v6m0 6v6M5.64 5.64l4.24 4.24m4.24 4.24l4.24 4.24M1 12h6m6 0h6M5.64 18.36l4.24-4.24m4.24-4.24l4.24-4.24"></path>
            </svg>
            <h3 className="card-title">Observability & Monitoring Demo</h3>
            <p className="card-description">
              Complete observability stack demonstration with Prometheus metrics
              collection, Grafana visualization, and Loki for log aggregation.
              Shows custom ML metrics tracking model performance, data drift
              detection, and alerting systems for ML applications.
            </p>
          </div>

          <div className="about-card">
            <svg
              className="card-icon"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z"></path>
              <path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z"></path>
              <path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0"></path>
              <path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5"></path>
            </svg>
            <h3 className="card-title">Containerized ML Architecture</h3>
            <p className="card-description">
              Demonstrates containerized deployment architecture with GPU
              acceleration support for model inference. Implements microservices
              pattern with dedicated vector database, search engine, and object
              storage for educational and research environments.
            </p>
          </div>

          <div className="about-card">
            <svg
              className="card-icon"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
              <path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
            </svg>
            <h3 className="card-title">Modern ML API Implementation</h3>
            <p className="card-description">
              Demonstrates async Python backend with FastAPI implementing
              RESTful APIs for ML model serving. Features automatic OpenAPI
              documentation, request validation, correlation tracking, and
              structured logging for concurrent ML inference workloads.
            </p>
          </div>
        </div>

        <div className="about-tech">
          <h3 className="tech-title">Complete Technical Stack</h3>
          <div className="tech-tags">
            <span className="tech-tag">FastAPI</span>
            <span className="tech-tag">Groq LLM</span>
            <span className="tech-tag">Transformers</span>
            <span className="tech-tag">CLIP Vision</span>
            <span className="tech-tag">Qdrant</span>
            <span className="tech-tag">Elasticsearch</span>
            <span className="tech-tag">Neo4j</span>
            <span className="tech-tag">Supabase</span>
            <span className="tech-tag">MinIO</span>
            <span className="tech-tag">Prometheus</span>
            <span className="tech-tag">Grafana</span>
            <span className="tech-tag">Loki</span>
            <span className="tech-tag">Docker</span>
            <span className="tech-tag">CUDA</span>
            <span className="tech-tag">React</span>
            <span className="tech-tag">TypeScript</span>
          </div>
        </div>
      </div>
    </section>
  );
};

export default About;
