import React from "react";
import Beams from "./Beams";
import "./Hero.css";

const Hero: React.FC = () => {
  const scrollToChat = () => {
    // Navigate to RAG page instead of scrolling to non-existent upload section
    window.location.href = "/rag";
  };

  return (
    <section id="hero" className="hero">
      <div className="hero-background">
        <Beams
          beamWidth={3}
          beamHeight={60}
          beamNumber={12}
          lightColor="#ffffff"
          speed={2}
          noiseIntensity={1.75}
          scale={0.2}
          rotation={45}
        />
      </div>
      <div className="hero-container">
        <div className="hero-content">
          <h1 className="hero-title">
            Enterprise-Grade Multimodal
            <br />
            <span className="hero-title-accent">RAG System</span>
          </h1>
          <div className="hero-actions">
            <button className="hero-button primary" onClick={scrollToChat}>
              Try Demo
            </button>
            <button
              className="hero-button secondary"
              onClick={() => {
                window.location.href = "/metrics";
              }}
            >
              View Metrics
            </button>
          </div>
          <div className="hero-features">
            <div className="hero-feature">
              <svg
                className="feature-icon"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
              </svg>
              <span className="feature-text">Ensemble Retrieval</span>
            </div>
            <div className="hero-feature">
              <svg
                className="feature-icon"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <circle cx="12" cy="12" r="10"></circle>
                <circle cx="12" cy="12" r="6"></circle>
                <circle cx="12" cy="12" r="2"></circle>
              </svg>
              <span className="feature-text">Full-Stack ML</span>
            </div>
            <div className="hero-feature">
              <svg
                className="feature-icon"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
                <path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
              </svg>
              <span className="feature-text">Research to Demo</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;
