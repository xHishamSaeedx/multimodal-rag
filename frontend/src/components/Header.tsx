import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import './Header.css';

const Header: React.FC = () => {
  const location = useLocation();

  return (
    <header className="header">
      <div className="header-container">
        <Link to="/" className="logo">
          <svg className="logo-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="11" cy="11" r="8"></circle>
            <path d="m21 21-4.35-4.35"></path>
          </svg>
          <span className="logo-text">Multimodal RAG</span>
        </Link>
        <nav className="nav">
          <Link
            to="/"
            className={`nav-link ${location.pathname === '/' ? 'active' : ''}`}
          >
            Home
          </Link>
          <Link
            to="/documents"
            className={`nav-link ${location.pathname === '/documents' ? 'active' : ''}`}
          >
            Documents
          </Link>
          <Link
            to="/chat"
            className={`nav-link ${location.pathname === '/chat' ? 'active' : ''}`}
          >
            Chat
          </Link>
        </nav>
      </div>
    </header>
  );
};

export default Header;

