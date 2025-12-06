import React from 'react';
import DocumentUpload from './DocumentUpload';
import DocumentList from './DocumentList';
import './Documents.css';

const Documents: React.FC = () => {
  return (
    <div className="documents-page">
      <div className="documents-container">
        <DocumentUpload />
        <DocumentList />
      </div>
    </div>
  );
};

export default Documents;
