import React from 'react';
import './Hero.css';

const Hero: React.FC = () => {
  const scrollToUpload = () => {
    const element = document.getElementById('upload');
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  return (
    <section id="hero" className="hero">
      <div className="hero-container">
        <div className="hero-content">
          <h1 className="hero-title">
            Hybrid Graph Multimodal
            <br />
            <span className="hero-title-accent">RAG System</span>
          </h1>
          <p className="hero-description">
            Enterprise-grade Retrieval-Augmented Generation system optimized for 
            large-scale, multimodal knowledge bases. Extract, index, and query 
            across text, images, tables, and diagrams with precision and speed.
          </p>
          <div className="hero-actions">
            <button className="hero-button primary" onClick={scrollToUpload}>
              Get Started
            </button>
            <button 
              className="hero-button secondary" 
              onClick={() => {
                const element = document.getElementById('about');
                if (element) {
                  element.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
              }}
            >
              Learn More
            </button>
          </div>
          <div className="hero-features">
            <div className="hero-feature">
              <svg className="feature-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
              </svg>
              <span className="feature-text">Fast Retrieval</span>
            </div>
            <div className="hero-feature">
              <svg className="feature-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10"></circle>
                <circle cx="12" cy="12" r="6"></circle>
                <circle cx="12" cy="12" r="2"></circle>
              </svg>
              <span className="feature-text">Multimodal</span>
            </div>
            <div className="hero-feature">
              <svg className="feature-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
                <path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
              </svg>
              <span className="feature-text">Enterprise-Grade</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;

