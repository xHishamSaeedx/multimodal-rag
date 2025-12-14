import axios from "axios";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

export interface IngestResponse {
  success: boolean;
  message: string;
  document_id?: string | null;
  object_key?: string | null; // MinIO object key (path) where document is stored
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

export interface SourceInfo {
  chunk_id: string;
  document_id: string;
  filename: string;
  chunk_index: number;
  chunk_text: string; // Truncated preview
  full_chunk_text: string; // Full chunk text
  citation: string;
  metadata?: Record<string, any> | null;
  image_path?: string | null; // Supabase storage path for image chunks
  image_url?: string | null; // Signed URL for image access (temporary)
}

export interface QueryRequest {
  query: string;
  limit?: number;
  include_sources?: boolean;
  filter_conditions?: Record<string, any>;
  enable_sparse?: boolean;
  enable_dense?: boolean;
  enable_table?: boolean;
  enable_image?: boolean;
  enable_graph?: boolean;
}

export interface QueryResponse {
  success: boolean;
  query: string;
  answer: string;
  sources: SourceInfo[];
  chunks_used: string[];
  model: string;
  tokens_used?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  } | null;
  retrieval_stats?: {
    chunks_found: number;
    chunks_used: number;
    retrieval_method: string;
  } | null;
}

export interface ModelsConfig {
  status: string;
  timestamp: string;
  models: {
    text_embedding: {
      model: string;
      dimension: number;
      device: string;
    };
    image_embedding: {
      model_type: string;
      model_name: string;
    };
    llm: {
      provider: string;
      model: string;
    };
    captioning: {
      model: string;
    };
    vision_processing: {
      mode: string;
    };
    retrieval_types: string[];
  };
}

export const api = {
  /**
   * Upload a single document for ingestion
   */
  uploadDocument: async (
    file: File,
    options?: {
      enableText?: boolean;
      enableTables?: boolean;
      enableImages?: boolean;
      enableGraph?: boolean;
    }
  ): Promise<IngestResponse> => {
    const formData = new FormData();
    formData.append("file", file);

    // Add processor configuration options
    // Explicitly convert boolean to string, defaulting to true if undefined
    // Ensure we always send 'true' or 'false' (lowercase strings) as expected by backend
    const enableText =
      options?.enableText !== undefined && options.enableText !== null
        ? String(options.enableText).toLowerCase()
        : "true";
    const enableTables =
      options?.enableTables !== undefined && options.enableTables !== null
        ? String(options.enableTables).toLowerCase()
        : "true";
    const enableImages =
      options?.enableImages !== undefined && options.enableImages !== null
        ? String(options.enableImages).toLowerCase()
        : "true";
    const enableGraph =
      options?.enableGraph !== undefined && options.enableGraph !== null
        ? String(options.enableGraph).toLowerCase()
        : "true";

    formData.append("enable_text", enableText);
    formData.append("enable_tables", enableTables);
    formData.append("enable_images", enableImages);
    formData.append("enable_graph", enableGraph);

    // Debug: log what we're sending (with original values for debugging)
    console.log("API uploadDocument - Options received:", {
      enableText: options?.enableText,
      enableTables: options?.enableTables,
      enableImages: options?.enableImages,
      enableGraph: options?.enableGraph,
    });
    console.log("API uploadDocument - Values being sent to backend:", {
      enableText,
      enableTables,
      enableImages,
      enableGraph,
    });

    const response = await apiClient.post<IngestResponse>(
      "/api/v1/ingest",
      formData,
      {
        headers: {
          "Content-Type": "multipart/form-data",
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
      formData.append("files", file);
    });

    const response = await apiClient.post<IngestResponse[]>(
      "/api/v1/ingest/batch",
      formData,
      {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      }
    );

    return response.data;
  },

  /**
   * Health check endpoint
   */
  healthCheck: async () => {
    const response = await apiClient.get("/api/v1/health");
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
      "/api/v1/documents",
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
        responseType: "blob",
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
    const link = document.createElement("a");
    link.href = url;
    link.download = filename || objectKey.split("/").pop() || "document";
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

  /**
   * Submit a query and get an answer from the RAG system
   */
  query: async (request: QueryRequest): Promise<QueryResponse> => {
    const response = await apiClient.post<QueryResponse>(
      "/api/v1/query",
      request
    );
    return response.data;
  },

  /**
   * Get a signed URL for an image
   */
  getImageUrl: async (
    imagePath: string,
    expiresIn: number = 3600
  ): Promise<string> => {
    // URL encode the image path
    const encodedPath = encodeURIComponent(imagePath);
    const response = await apiClient.get<{
      success: boolean;
      image_url: string;
      expires_in: number;
      image_path: string;
    }>(`/api/v1/images/${encodedPath}/url`, {
      params: { expires_in: expiresIn },
    });
    return response.data.image_url;
  },

  /**
   * Get current models configuration
   */
  getCurrentModels: async (): Promise<ModelsConfig> => {
    const response = await apiClient.get<ModelsConfig>("/api/v1/health/models");
    return response.data;
  },
};

export default api;
