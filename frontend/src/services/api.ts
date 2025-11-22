import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface IngestResponse {
  success: boolean;
  message: string;
  document_id?: string | null;
  object_key?: string | null;  // MinIO object key (path) where document is stored
  file_name: string;
  file_type: string;
  file_size: number;
  page_count?: number | null;
  extracted_text_length: number;
  metadata?: Record<string, any> | null;
  extracted_at: string;
}

export interface ErrorResponse {
  success: false;
  error: string;
  details?: Record<string, any>;
}

export interface DocumentInfo {
  object_key: string;
  filename: string;
  size: number;
  last_modified?: string | null;
  content_type: string;
  source: string;
  document_type: string;
}

export interface DocumentListResponse {
  success: boolean;
  documents: DocumentInfo[];
  count: number;
}

export interface DocumentDeleteResponse {
  success: boolean;
  message: string;
  object_key: string;
}

export const api = {
  /**
   * Upload a single document for ingestion
   */
  uploadDocument: async (file: File): Promise<IngestResponse> => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await apiClient.post<IngestResponse>(
      '/api/v1/ingest',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );

    return response.data;
  },

  /**
   * Upload multiple documents for batch ingestion
   */
  uploadDocuments: async (files: File[]): Promise<IngestResponse[]> => {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file);
    });

    const response = await apiClient.post<IngestResponse[]>(
      '/api/v1/ingest/batch',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );

    return response.data;
  },

  /**
   * Health check endpoint
   */
  healthCheck: async () => {
    const response = await apiClient.get('/api/v1/health');
    return response.data;
  },

  /**
   * List all documents
   */
  listDocuments: async (params?: {
    prefix?: string;
    source?: string;
    document_type?: string;
    limit?: number;
  }): Promise<DocumentListResponse> => {
    const response = await apiClient.get<DocumentListResponse>(
      '/api/v1/documents',
      { params }
    );
    return response.data;
  },

  /**
   * Get/download a document by object key
   */
  getDocument: async (
    objectKey: string,
    download: boolean = false
  ): Promise<Blob> => {
    // URL encode the object key
    const encodedKey = encodeURIComponent(objectKey);
    const response = await apiClient.get<Blob>(
      `/api/v1/documents/${encodedKey}`,
      {
        params: { download },
        responseType: 'blob',
      }
    );
    return response.data;
  },

  /**
   * Download a document as a file
   */
  downloadDocument: async (
    objectKey: string,
    filename?: string
  ): Promise<void> => {
    const blob = await api.getDocument(objectKey, true);
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename || objectKey.split('/').pop() || 'document';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  },

  /**
   * Delete a document by object key
   */
  deleteDocument: async (
    objectKey: string
  ): Promise<DocumentDeleteResponse> => {
    // URL encode the object key
    const encodedKey = encodeURIComponent(objectKey);
    const response = await apiClient.delete<DocumentDeleteResponse>(
      `/api/v1/documents/${encodedKey}`
    );
    return response.data;
  },
};

export default api;

