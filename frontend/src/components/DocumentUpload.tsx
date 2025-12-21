import React, {
  useState,
  useCallback,
  useRef,
  useEffect,
  forwardRef,
  useImperativeHandle,
} from "react";
import { api, IngestResponse } from "../services/api";

interface UploadResult extends IngestResponse {
  file?: File;
}

export interface DocumentUploadHandle {
  handleFiles: (files: FileList | null) => void;
}

const DocumentUpload = forwardRef<DocumentUploadHandle>((props, ref) => {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadResults, setUploadResults] = useState<UploadResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Processor configuration state
  const [enableText, setEnableText] = useState(true);
  const [enableTables, setEnableTables] = useState(true);
  const [enableImages, setEnableImages] = useState(true);
  const [enableGraph, setEnableGraph] = useState(true);

  // Refs to always get the latest state values (avoid stale closures)
  const enableTextRef = useRef(enableText);
  const enableTablesRef = useRef(enableTables);
  const enableImagesRef = useRef(enableImages);
  const enableGraphRef = useRef(enableGraph);

  // Keep refs in sync with state
  useEffect(() => {
    enableTextRef.current = enableText;
    console.log("enableText state changed to:", enableText);
  }, [enableText]);

  useEffect(() => {
    enableTablesRef.current = enableTables;
    console.log("enableTables state changed to:", enableTables);
  }, [enableTables]);

  useEffect(() => {
    enableImagesRef.current = enableImages;
    console.log("enableImages state changed to:", enableImages);
  }, [enableImages]);

  useEffect(() => {
    enableGraphRef.current = enableGraph;
    console.log("enableGraph state changed to:", enableGraph);
  }, [enableGraph]);

  // Log initial state on mount
  useEffect(() => {
    console.log("DocumentUpload mounted with initial state:", {
      enableText,
      enableTables,
      enableImages,
      enableGraph,
    });
  }, []); // Run once on mount

  const supportedTypes = [".pdf", ".docx", ".txt", ".md", ".markdown"];
  const maxFileSize = 50 * 1024 * 1024; // 50MB

  const validateFile = (file: File): string | null => {
    const ext = "." + file.name.split(".").pop()?.toLowerCase();

    if (!supportedTypes.includes(ext)) {
      return `Unsupported file type. Supported types: ${supportedTypes.join(
        ", "
      )}`;
    }

    if (file.size > maxFileSize) {
      return `File size exceeds maximum of ${maxFileSize / (1024 * 1024)}MB`;
    }

    return null;
  };

  const uploadFile = useCallback(async (file: File) => {
    const validationError = validateFile(file);
    if (validationError) {
      throw new Error(validationError);
    }

    // Use refs to get the latest state values (avoid stale closures)
    const currentEnableText = enableTextRef.current;
    const currentEnableTables = enableTablesRef.current;
    const currentEnableImages = enableImagesRef.current;
    const currentEnableGraph = enableGraphRef.current;

    // Debug: Log current state values from refs (latest values)
    console.log("DocumentUpload.uploadFile - Reading values from refs:", {
      enableText: currentEnableText,
      enableTables: currentEnableTables,
      enableImages: currentEnableImages,
      enableGraph: currentEnableGraph,
    });

    // Also log actual state values for comparison
    console.log("DocumentUpload.uploadFile - Actual state values:", {
      enableText,
      enableTables,
      enableImages,
      enableGraph,
    });

    // Safety check: If all processors are disabled, at least enable text processing
    // (This prevents accidentally uploading with no content extraction)
    let finalEnableText = currentEnableText;
    let finalEnableTables = currentEnableTables;
    let finalEnableImages = currentEnableImages;

    if (!currentEnableText && !currentEnableTables && !currentEnableImages) {
      console.warn(
        "All processors disabled! Auto-enabling text processing for basic functionality."
      );
      finalEnableText = true;
    }

    try {
      console.log(
        "DocumentUpload.uploadFile - Calling api.uploadDocument with:",
        {
          enableText: finalEnableText,
          enableTables: finalEnableTables,
          enableImages: finalEnableImages,
          enableGraph: currentEnableGraph,
        }
      );

      const result = await api.uploadDocument(file, {
        enableText: finalEnableText,
        enableTables: finalEnableTables,
        enableImages: finalEnableImages,
        enableGraph: currentEnableGraph,
      });
      return { ...result, file };
    } catch (err: any) {
      const errorMessage =
        err.response?.data?.error ||
        err.response?.data?.detail?.error ||
        err.message ||
        "Upload failed";
      throw new Error(errorMessage);
    }
  }, []); // Empty deps - we use refs to get latest values

  const handleFiles = useCallback(
    async (files: FileList | null) => {
      if (!files || files.length === 0) return;

      setIsUploading(true);
      setError(null);
      const newResults: UploadResult[] = [];

      try {
        // Process files sequentially to avoid overwhelming the server
        for (let i = 0; i < files.length; i++) {
          const file = files[i];
          try {
            const result = await uploadFile(file);
            newResults.push(result);
            setUploadResults((prev) => [...prev, result]);
          } catch (err: any) {
            const errorResult: UploadResult = {
              success: false,
              message: err.message || "Upload failed",
              file_name: file.name,
              file_type: "unknown",
              file_size: file.size,
              extracted_text_length: 0,
              extracted_at: new Date().toISOString(),
              file,
            };
            newResults.push(errorResult);
            setUploadResults((prev) => [...prev, errorResult]);
          }
        }
      } catch (err: any) {
        setError(err.message || "An unexpected error occurred");
      } finally {
        setIsUploading(false);
      }
    },
    [uploadFile]
  );

  // Expose handleFiles to parent component via ref
  useImperativeHandle(ref, () => ({
    handleFiles,
  }));

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      handleFiles(e.target.files);
      // Reset input to allow re-uploading the same file
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    },
    [handleFiles]
  );

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + " " + sizes[i];
  };

  const formatDate = (dateString: string): string => {
    return new Date(dateString).toLocaleString();
  };

  const clearResults = () => {
    setUploadResults([]);
    setError(null);
  };

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const target = e.currentTarget as HTMLElement;
    const relatedTarget = e.relatedTarget as HTMLElement | null;
    if (!relatedTarget || !target.contains(relatedTarget)) {
      setIsDragging(false);
    }
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);
      if (!isUploading) {
        handleFiles(e.dataTransfer.files);
      }
    },
    [isUploading, handleFiles]
  );

  return (
    <div className="document-upload-container">
      <div className="processor-config-wrapper">
        <div className="processor-config">
          <div className="processor-header">
            <h3>Document Processors</h3>
            <p className="processor-description">
              Select which content types to process from your documents.
              Knowledge Graph builds relationships between extracted content.
            </p>
            {isUploading && (
              <div className="upload-status-inline">
                <div className="spinner"></div>
                <span>Uploading and processing...</span>
              </div>
            )}
            {error && (
              <div className="error-message-inline">
                <strong>Error:</strong> {error}
              </div>
            )}
            {!enableText && !enableTables && !enableImages && (
              <div
                style={{
                  marginTop: "10px",
                  padding: "10px",
                  backgroundColor: enableGraph ? "#fff3cd" : "#f8d7da",
                  border: `1px solid ${enableGraph ? "#ffc107" : "#dc3545"}`,
                  borderRadius: "4px",
                  color: enableGraph ? "#856404" : "#721c24",
                }}
              >
                <strong>⚠️ {enableGraph ? "Warning" : "Error"}:</strong> All
                content processors (Text, Tables, Images) are disabled. No
                content will be extracted or indexed.
                {enableGraph && (
                  <div
                    style={{
                      marginTop: "8px",
                      paddingTop: "8px",
                      borderTop: "1px solid #ffc107",
                    }}
                  >
                    <strong>⚠️ Important:</strong> Knowledge Graph is currently{" "}
                    <strong>enabled</strong>, but it
                    <strong>
                      {" "}
                      requires at least one content processor
                    </strong>{" "}
                    (Text, Tables, or Images) to extract content before it can
                    build relationships. Please enable at least one content
                    processor above.
                  </div>
                )}
                {!enableGraph && (
                  <div style={{ marginTop: "8px" }}>
                    <strong>Tip:</strong> Enable at least one content processor
                    (Text, Tables, or Images) to extract content from your
                    documents.
                  </div>
                )}
              </div>
            )}
          </div>
          <div className="processor-options">
            <label className={`processor-option ${enableText ? "active" : ""}`}>
              <div className="processor-checkbox-wrapper">
                <input
                  type="checkbox"
                  checked={enableText}
                  onChange={(e) => {
                    console.log("Text checkbox changed:", e.target.checked);
                    setEnableText(e.target.checked);
                  }}
                  disabled={isUploading}
                />
                <span className="processor-label">Text</span>
              </div>
            </label>
            <label
              className={`processor-option ${enableTables ? "active" : ""}`}
            >
              <div className="processor-checkbox-wrapper">
                <input
                  type="checkbox"
                  checked={enableTables}
                  onChange={(e) => {
                    console.log("Tables checkbox changed:", e.target.checked);
                    setEnableTables(e.target.checked);
                  }}
                  disabled={isUploading}
                />
                <span className="processor-label">Tables</span>
              </div>
            </label>
            <label
              className={`processor-option ${enableImages ? "active" : ""}`}
            >
              <div className="processor-checkbox-wrapper">
                <input
                  type="checkbox"
                  checked={enableImages}
                  onChange={(e) => {
                    console.log("Images checkbox changed:", e.target.checked);
                    setEnableImages(e.target.checked);
                  }}
                  disabled={isUploading}
                />
                <span className="processor-label">Images</span>
              </div>
            </label>
            <label
              className={`processor-option ${enableGraph ? "active" : ""}`}
            >
              <div className="processor-checkbox-wrapper">
                <input
                  type="checkbox"
                  checked={enableGraph}
                  onChange={(e) => {
                    console.log(
                      "Knowledge Graph checkbox changed:",
                      e.target.checked
                    );
                    setEnableGraph(e.target.checked);
                  }}
                  disabled={isUploading}
                />
                <span className="processor-label">Knowledge Graph</span>
              </div>
            </label>
          </div>
        </div>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept={supportedTypes.join(",")}
        onChange={handleFileInput}
        className="file-input"
        disabled={isUploading}
      />

      {/* Drag and Drop Zone */}
      <div className="upload-zone-wrapper">
        <div
          className={`upload-zone ${isUploading ? "uploading" : ""} ${
            isDragging ? "dragging" : ""
          }`}
          role="button"
          tabIndex={0}
          onClick={() => {
            if (!isUploading && fileInputRef.current) {
              fileInputRef.current.click();
            }
          }}
          onKeyDown={(e) => {
            if (
              (e.key === "Enter" || e.key === " ") &&
              !isUploading &&
              fileInputRef.current
            ) {
              e.preventDefault();
              fileInputRef.current.click();
            }
          }}
          onDragEnter={handleDragEnter}
          onDragLeave={handleDragLeave}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          aria-label="Upload documents by clicking or dragging files here"
        >
          <div className="upload-content">
            <svg
              className="upload-icon"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="17 8 12 3 7 8"></polyline>
              <line x1="12" y1="3" x2="12" y2="15"></line>
            </svg>
            <p className="upload-text">Drag and drop documents here</p>
            <button
              type="button"
              className="choose-files-button"
              onClick={(e) => {
                e.stopPropagation();
                if (!isUploading && fileInputRef.current) {
                  fileInputRef.current.click();
                }
              }}
              disabled={isUploading}
            >
              Choose Files
            </button>
            <p className="upload-hint">
              Supported formats: PDF, DOCX, TXT, MD, MARKDOWN (Max 50MB)
            </p>
          </div>
        </div>
      </div>

      {uploadResults.length > 0 && (
        <div className="results-section">
          <div className="results-header">
            <h2>Upload Results</h2>
            <button
              type="button"
              className="clear-button"
              onClick={clearResults}
            >
              Clear
            </button>
          </div>

          <div className="results-list">
            {uploadResults.map((result, index) => (
              <div
                key={index}
                className={`result-card ${
                  result.success ? "success" : "error"
                }`}
              >
                <div className="result-header">
                  <div className="result-status">
                    {result.success ? (
                      <svg
                        className="status-icon success"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                      >
                        <polyline points="20 6 9 17 4 12"></polyline>
                      </svg>
                    ) : (
                      <svg
                        className="status-icon error"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                      >
                        <line x1="18" y1="6" x2="6" y2="18"></line>
                        <line x1="6" y1="6" x2="18" y2="18"></line>
                      </svg>
                    )}
                    <span className="file-name">{result.file_name}</span>
                  </div>
                  <span
                    className={`status-badge ${
                      result.success ? "success" : "error"
                    }`}
                  >
                    {result.success ? "Success" : "Failed"}
                  </span>
                </div>

                <div className="result-details">
                  <p className="result-message">{result.message}</p>

                  {result.success && (
                    <div className="result-meta">
                      <div className="meta-item">
                        <strong>Type:</strong> {result.file_type.toUpperCase()}
                      </div>
                      <div className="meta-item">
                        <strong>Size:</strong>{" "}
                        {formatFileSize(result.file_size)}
                      </div>
                      {result.page_count && (
                        <div className="meta-item">
                          <strong>Pages:</strong> {result.page_count}
                        </div>
                      )}
                      <div className="meta-item">
                        <strong>Text Length:</strong>{" "}
                        {result.extracted_text_length.toLocaleString()}{" "}
                        characters
                      </div>
                      <div className="meta-item">
                        <strong>Extracted At:</strong>{" "}
                        {formatDate(result.extracted_at)}
                      </div>
                    </div>
                  )}

                  {result.metadata &&
                    Object.keys(result.metadata).length > 0 && (
                      <details className="metadata-details">
                        <summary>Metadata</summary>
                        <pre className="metadata-content">
                          {JSON.stringify(result.metadata, null, 2)}
                        </pre>
                      </details>
                    )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
});

DocumentUpload.displayName = "DocumentUpload";

export default DocumentUpload;
