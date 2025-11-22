import React from 'react';
import './About.css';

const About: React.FC = () => {
  return (
    <section id="about" className="about">
      <div className="about-container">
        <div className="about-header">
          <h2 className="about-title">Enterprise-Grade Architecture</h2>
          <p className="about-subtitle">
            A next-generation RAG system designed for speed, accuracy, and scalability
          </p>
        </div>

        <div className="about-grid">
          <div className="about-card">
            <svg className="card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="12" y1="20" x2="12" y2="10"></line>
              <line x1="18" y1="20" x2="18" y2="4"></line>
              <line x1="6" y1="20" x2="6" y2="16"></line>
            </svg>
            <h3 className="card-title">Hybrid Retrieval</h3>
            <p className="card-description">
              Combines BM25 sparse search, dense vector embeddings, and metadata 
              filtering for optimal retrieval accuracy across diverse query types.
            </p>
          </div>

          <div className="about-card">
            <svg className="card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="3" y="3" width="7" height="7"></rect>
              <rect x="14" y="3" width="7" height="7"></rect>
              <rect x="14" y="14" width="7" height="7"></rect>
              <rect x="3" y="14" width="7" height="7"></rect>
            </svg>
            <h3 className="card-title">Multimodal Understanding</h3>
            <p className="card-description">
              Process and understand text, tables, images, diagrams, and OCR content 
              in a unified pipeline with intelligent chunking strategies.
            </p>
          </div>

          <div className="about-card">
            <svg className="card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"></path>
              <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"></path>
            </svg>
            <h3 className="card-title">Knowledge Graph</h3>
            <p className="card-description">
              Graph-based reasoning for complex multi-hop queries requiring 
              relational understanding across entities and documents.
            </p>
          </div>

          <div className="about-card">
            <svg className="card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="3"></circle>
              <path d="M12 1v6m0 6v6M5.64 5.64l4.24 4.24m4.24 4.24l4.24 4.24M1 12h6m6 0h6M5.64 18.36l4.24-4.24m4.24-4.24l4.24-4.24"></path>
            </svg>
            <h3 className="card-title">MLOps Pipeline</h3>
            <p className="card-description">
              Full continuous improvement pipeline with A/B testing, feedback loops, 
              and automated model updates for production deployments.
            </p>
          </div>

          <div className="about-card">
            <svg className="card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z"></path>
              <path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z"></path>
              <path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0"></path>
              <path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5"></path>
            </svg>
            <h3 className="card-title">Scalable Infrastructure</h3>
            <p className="card-description">
              Built for enterprise scale with microservices architecture, distributed 
              indexing, and horizontal scalability across cloud platforms.
            </p>
          </div>

          <div className="about-card">
            <svg className="card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
              <path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
            </svg>
            <h3 className="card-title">Secure & Compliant</h3>
            <p className="card-description">
              Enterprise-grade security with ACLs, version control, audit logging, 
              and compliance-ready metadata management.
            </p>
          </div>
        </div>

        <div className="about-tech">
          <h3 className="tech-title">Supported File Types</h3>
          <div className="tech-tags">
            <span className="tech-tag">PDF</span>
            <span className="tech-tag">DOCX</span>
            <span className="tech-tag">TXT</span>
            <span className="tech-tag">Markdown</span>
            <span className="tech-tag">Images</span>
            <span className="tech-tag">Tables</span>
            <span className="tech-tag">Diagrams</span>
          </div>
        </div>
      </div>
    </section>
  );
};

export default About;

