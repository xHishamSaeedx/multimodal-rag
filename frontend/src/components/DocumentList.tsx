import React, { useState, useEffect } from 'react';
import { api, DocumentInfo } from '../services/api';
import './DocumentList.css';

const DocumentList: React.FC = () => {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filters, setFilters] = useState<{
    source?: string;
    document_type?: string;
  }>({});
  const [deletingKeys, setDeletingKeys] = useState<Set<string>>(new Set());

  useEffect(() => {
    loadDocuments();
  }, [filters]);

  const loadDocuments = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await api.listDocuments(filters);
      setDocuments(response.documents);
    } catch (err: any) {
      const errorMessage = 
        err.response?.data?.error || 
        err.response?.data?.detail?.error || 
        err.message || 
        'Failed to load documents';
      setError(errorMessage);
      console.error('Error loading documents:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDelete = async (objectKey: string, filename: string) => {
    if (!window.confirm(`Are you sure you want to delete "${filename}"?`)) {
      return;
    }

    setDeletingKeys((prev) => new Set(prev).add(objectKey));
    try {
      await api.deleteDocument(objectKey);
      // Remove from local state
      setDocuments((prev) => prev.filter((doc) => doc.object_key !== objectKey));
    } catch (err: any) {
      const errorMessage = 
        err.response?.data?.error || 
        err.response?.data?.detail?.error || 
        err.message || 
        'Failed to delete document';
      alert(`Error deleting document: ${errorMessage}`);
      console.error('Error deleting document:', err);
    } finally {
      setDeletingKeys((prev) => {
        const newSet = new Set(prev);
        newSet.delete(objectKey);
        return newSet;
      });
    }
  };

  const handleDownload = async (objectKey: string, filename: string) => {
    try {
      await api.downloadDocument(objectKey, filename);
    } catch (err: any) {
      const errorMessage = 
        err.response?.data?.error || 
        err.response?.data?.detail?.error || 
        err.message || 
        'Failed to download document';
      alert(`Error downloading document: ${errorMessage}`);
      console.error('Error downloading document:', err);
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  const formatDate = (dateString: string | null | undefined): string => {
    if (!dateString) return 'Unknown';
    return new Date(dateString).toLocaleString();
  };

  const getFileIcon = (fileType: string): string => {
    switch (fileType.toLowerCase()) {
      case 'pdf':
        return '📄';
      case 'docx':
      case 'doc':
        return '📝';
      case 'txt':
      case 'md':
      case 'markdown':
        return '📄';
      default:
        return '📎';
    }
  };

  // Get unique sources and document types for filters
  const uniqueSources = Array.from(new Set(documents.map((doc) => doc.source))).sort();
  const uniqueTypes = Array.from(new Set(documents.map((doc) => doc.document_type))).sort();

  return (
    <div className="document-list-container">
      <div className="document-list-header">
        <h1>Uploaded Documents</h1>
        <button 
          className="refresh-button" 
          onClick={loadDocuments}
          disabled={isLoading}
          title="Refresh list"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="23 4 23 10 17 10"></polyline>
            <polyline points="1 20 1 14 7 14"></polyline>
            <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path>
          </svg>
          Refresh
        </button>
      </div>

      {/* Filters */}
      <div className="filters-section">
        <div className="filter-group">
          <label htmlFor="source-filter">Source:</label>
          <select
            id="source-filter"
            value={filters.source || ''}
            onChange={(e) =>
              setFilters((prev) => ({
                ...prev,
                source: e.target.value || undefined,
              }))
            }
          >
            <option value="">All Sources</option>
            {uniqueSources.map((source) => (
              <option key={source} value={source}>
                {source}
              </option>
            ))}
          </select>
        </div>

        <div className="filter-group">
          <label htmlFor="type-filter">Type:</label>
          <select
            id="type-filter"
            value={filters.document_type || ''}
            onChange={(e) =>
              setFilters((prev) => ({
                ...prev,
                document_type: e.target.value || undefined,
              }))
            }
          >
            <option value="">All Types</option>
            {uniqueTypes.map((type) => (
              <option key={type} value={type}>
                {type.toUpperCase()}
              </option>
            ))}
          </select>
        </div>

        {(filters.source || filters.document_type) && (
          <button
            className="clear-filters-button"
            onClick={() => setFilters({})}
          >
            Clear Filters
          </button>
        )}
      </div>

      {error && (
        <div className="error-message">
          <strong>Error:</strong> {error}
          <button onClick={loadDocuments} className="retry-button">
            Retry
          </button>
        </div>
      )}

      {isLoading && documents.length === 0 ? (
        <div className="loading-container">
          <div className="spinner"></div>
          <p>Loading documents...</p>
        </div>
      ) : documents.length === 0 ? (
        <div className="empty-state">
          <svg className="empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
            <polyline points="14 2 14 8 20 8"></polyline>
            <line x1="16" y1="13" x2="8" y2="13"></line>
            <line x1="16" y1="17" x2="8" y2="17"></line>
          </svg>
          <p>No documents found</p>
          <p className="empty-hint">
            {filters.source || filters.document_type
              ? 'Try adjusting your filters'
              : 'Upload documents to get started'}
          </p>
        </div>
      ) : (
        <>
          <div className="document-count">
            Showing {documents.length} document{documents.length !== 1 ? 's' : ''}
          </div>
          <div className="documents-grid">
            {documents.map((doc) => (
              <div key={doc.object_key} className="document-card">
                <div className="document-card-header">
                  <div className="document-icon">
                    {getFileIcon(doc.document_type)}
                  </div>
                  <div className="document-title-group">
                    <h3 className="document-title" title={doc.filename}>
                      {doc.filename}
                    </h3>
                    <div className="document-meta">
                      <span className="document-type">{doc.document_type.toUpperCase()}</span>
                      <span className="document-source">{doc.source}</span>
                    </div>
                  </div>
                </div>

                <div className="document-details">
                  <div className="detail-row">
                    <span className="detail-label">Size:</span>
                    <span className="detail-value">{formatFileSize(doc.size)}</span>
                  </div>
                  <div className="detail-row">
                    <span className="detail-label">Uploaded:</span>
                    <span className="detail-value">{formatDate(doc.last_modified)}</span>
                  </div>
                </div>

                <div className="document-actions">
                  <button
                    className="action-button download-button"
                    onClick={() => handleDownload(doc.object_key, doc.filename)}
                    title="Download document"
                  >
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                      <polyline points="7 10 12 15 17 10"></polyline>
                      <line x1="12" y1="15" x2="12" y2="3"></line>
                    </svg>
                    Download
                  </button>
                  <button
                    className="action-button delete-button"
                    onClick={() => handleDelete(doc.object_key, doc.filename)}
                    disabled={deletingKeys.has(doc.object_key)}
                    title="Delete document"
                  >
                    {deletingKeys.has(doc.object_key) ? (
                      <div className="button-spinner"></div>
                    ) : (
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <polyline points="3 6 5 6 21 6"></polyline>
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                        <line x1="10" y1="11" x2="10" y2="17"></line>
                        <line x1="14" y1="11" x2="14" y2="17"></line>
                      </svg>
                    )}
                    {deletingKeys.has(doc.object_key) ? 'Deleting...' : 'Delete'}
                  </button>
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
};

export default DocumentList;

