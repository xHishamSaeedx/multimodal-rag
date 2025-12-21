import React, { useState, useEffect } from "react";
import { api, DocumentInfo } from "../services/api";

const DocumentList: React.FC = () => {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [allDocuments, setAllDocuments] = useState<DocumentInfo[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [deletingKeys, setDeletingKeys] = useState<Set<string>>(new Set());

  useEffect(() => {
    loadDocuments();
  }, []);

  useEffect(() => {
    if (searchQuery.trim() === "") {
      setDocuments(allDocuments);
    } else {
      const query = searchQuery.toLowerCase().trim();
      const filtered = allDocuments.filter(
        (doc) =>
          doc.filename.toLowerCase().includes(query) ||
          doc.source.toLowerCase().includes(query) ||
          doc.document_type.toLowerCase().includes(query)
      );
      setDocuments(filtered);
    }
  }, [searchQuery, allDocuments]);

  const loadDocuments = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await api.listDocuments({});
      setAllDocuments(response.documents);
      setDocuments(response.documents);
    } catch (err: any) {
      const errorMessage =
        err.response?.data?.error ||
        err.response?.data?.detail?.error ||
        err.message ||
        "Failed to load documents";
      setError(errorMessage);
      console.error("Error loading documents:", err);
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
      setDocuments((prev) =>
        prev.filter((doc) => doc.object_key !== objectKey)
      );
    } catch (err: any) {
      const errorMessage =
        err.response?.data?.error ||
        err.response?.data?.detail?.error ||
        err.message ||
        "Failed to delete document";
      alert(`Error deleting document: ${errorMessage}`);
      console.error("Error deleting document:", err);
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
        "Failed to download document";
      alert(`Error downloading document: ${errorMessage}`);
      console.error("Error downloading document:", err);
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + " " + sizes[i];
  };

  const formatDate = (dateString: string | null | undefined): string => {
    if (!dateString) return "Unknown";
    return new Date(dateString).toLocaleString();
  };

  const getFileIcon = (fileType: string): string => {
    switch (fileType.toLowerCase()) {
      case "pdf":
        return "📄";
      case "docx":
      case "doc":
        return "📝";
      case "txt":
      case "md":
      case "markdown":
        return "📄";
      default:
        return "📎";
    }
  };

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
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <polyline points="23 4 23 10 17 10"></polyline>
            <polyline points="1 20 1 14 7 14"></polyline>
            <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path>
          </svg>
          Refresh
        </button>
      </div>

      {/* Search Bar */}
      <div className="search-section">
        <div className="search-input-wrapper">
          <svg
            className="search-icon"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <circle cx="11" cy="11" r="8"></circle>
            <path d="m21 21-4.35-4.35"></path>
          </svg>
          <input
            type="text"
            className="search-input"
            placeholder="Search documents..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          {searchQuery && (
            <button
              className="clear-search-button"
              onClick={() => setSearchQuery("")}
              title="Clear search"
            >
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <line x1="18" y1="6" x2="6" y2="18"></line>
                <line x1="6" y1="6" x2="18" y2="18"></line>
              </svg>
            </button>
          )}
        </div>
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
          <svg
            className="empty-icon"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
            <polyline points="14 2 14 8 20 8"></polyline>
            <line x1="16" y1="13" x2="8" y2="13"></line>
            <line x1="16" y1="17" x2="8" y2="17"></line>
          </svg>
          <p>No documents found</p>
          <p className="empty-hint">
            {searchQuery.trim()
              ? "Try a different search term"
              : "Upload documents to get started"}
          </p>
        </div>
      ) : (
        <>
          <div className="document-count">
            {searchQuery.trim() ? (
              <>
                Showing {documents.length} of {allDocuments.length} document
                {allDocuments.length !== 1 ? "s" : ""}
              </>
            ) : (
              <>
                Showing {documents.length} document
                {documents.length !== 1 ? "s" : ""}
              </>
            )}
          </div>
          <div className="documents-list-container">
            <div className="documents-list">
              {documents.map((doc) => (
                <div key={doc.object_key} className="document-list-item">
                  <div className="document-list-item-content">
                    <div className="document-list-item-icon">
                      {getFileIcon(doc.document_type)}
                    </div>
                    <div className="document-list-item-info">
                      <div
                        className="document-list-item-title"
                        title={doc.filename}
                      >
                        {doc.filename}
                      </div>
                      <div className="document-list-item-meta">
                        <span className="document-list-item-source">
                          {doc.source}
                        </span>
                        <span className="document-list-item-separator">•</span>
                        <span className="document-list-item-size">
                          {formatFileSize(doc.size)}
                        </span>
                        <span className="document-list-item-separator">•</span>
                        <span className="document-list-item-date">
                          {formatDate(doc.last_modified)}
                        </span>
                      </div>
                    </div>
                  </div>
                  <div className="document-list-item-actions">
                    <button
                      className="document-list-action-button download-button"
                      onClick={() =>
                        handleDownload(doc.object_key, doc.filename)
                      }
                      title="Download document"
                    >
                      <svg
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                      >
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                        <polyline points="7 10 12 15 17 10"></polyline>
                        <line x1="12" y1="15" x2="12" y2="3"></line>
                      </svg>
                    </button>
                    <button
                      className="document-list-action-button delete-button"
                      onClick={() => handleDelete(doc.object_key, doc.filename)}
                      disabled={deletingKeys.has(doc.object_key)}
                      title="Delete document"
                    >
                      {deletingKeys.has(doc.object_key) ? (
                        <div className="button-spinner"></div>
                      ) : (
                        <svg
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2"
                        >
                          <polyline points="3 6 5 6 21 6"></polyline>
                          <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                          <line x1="10" y1="11" x2="10" y2="17"></line>
                          <line x1="14" y1="11" x2="14" y2="17"></line>
                        </svg>
                      )}
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default DocumentList;
