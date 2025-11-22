import React from 'react';
import Header from './components/Header';
import Hero from './components/Hero';
import About from './components/About';
import DocumentUpload from './components/DocumentUpload';
import DocumentList from './components/DocumentList';
import './App.css';

function App() {
  return (
    <div className="App">
      <Header />
      <Hero />
      <About />
      <section id="upload" className="upload-section">
        <DocumentUpload />
      </section>
      <section id="documents" className="documents-section">
        <DocumentList />
      </section>
    </div>
  );
}

export default App;

