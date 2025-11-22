import React from 'react';
import './Header.css';

const Header: React.FC = () => {
  const scrollToSection = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  return (
    <header className="header">
      <div className="header-container">
        <div className="logo">
          <svg className="logo-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="11" cy="11" r="8"></circle>
            <path d="m21 21-4.35-4.35"></path>
          </svg>
          <span className="logo-text">Multimodal RAG</span>
        </div>
        <nav className="nav">
          <button 
            className="nav-link" 
            onClick={() => scrollToSection('hero')}
          >
            Home
          </button>
          <button 
            className="nav-link" 
            onClick={() => scrollToSection('about')}
          >
            About
          </button>
          <button 
            className="nav-link" 
            onClick={() => scrollToSection('upload')}
          >
            Upload
          </button>
          <button 
            className="nav-link" 
            onClick={() => scrollToSection('documents')}
          >
            Documents
          </button>
        </nav>
      </div>
    </header>
  );
};

export default Header;

