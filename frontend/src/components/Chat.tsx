import React, { useState, useRef, useEffect, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { api, QueryResponse, SourceInfo } from '../services/api';
import './Chat.css';

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  sources?: SourceInfo[];
  timestamp: Date;
}

interface ImageState {
  url: string;
  loading: boolean;
  error: boolean;
  expiresAt: number; // Timestamp when URL expires
}

const Chat: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedImage, setExpandedImage] = useState<{ url: string; alt: string } | null>(null);
  const [imageStates, setImageStates] = useState<Map<string, ImageState>>(new Map());
  const [expandedSources, setExpandedSources] = useState<Set<string>>(new Set());
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const refreshTimersRef = useRef<Map<string, NodeJS.Timeout>>(new Map());
  
  // Retriever configuration state
  const [enableSparse, setEnableSparse] = useState(true);
  const [enableDense, setEnableDense] = useState(true);
  const [enableTable, setEnableTable] = useState(true);
  const [enableImage, setEnableImage] = useState(true);
  const [enableGraph, setEnableGraph] = useState(true);


  useEffect(() => {
    // Add welcome message
    setMessages([
      {
        id: 'welcome',
        type: 'assistant',
        content: 'Hello! I\'m your RAG assistant. Ask me anything about the documents you\'ve uploaded, and I\'ll find the relevant information for you.',
        timestamp: new Date(),
      },
    ]);
  }, []);

  // Cleanup timers on unmount
  useEffect(() => {
    return () => {
      refreshTimersRef.current.forEach((timer) => clearTimeout(timer));
      refreshTimersRef.current.clear();
    };
  }, []);

  // Refresh image URL before expiration (refresh at 80% of expiration time)
  const scheduleImageRefresh = useCallback((imagePath: string, expiresIn: number) => {
    // Clear existing timer if any
    const existingTimer = refreshTimersRef.current.get(imagePath);
    if (existingTimer) {
      clearTimeout(existingTimer);
    }

    // Schedule refresh at 80% of expiration time
    const refreshDelay = (expiresIn * 0.8) * 1000; // Convert to milliseconds
    
    const timer = setTimeout(async () => {
      try {
        const newUrl = await api.getImageUrl(imagePath, 3600);
        setImageStates((prev) => {
          const newMap = new Map(prev);
          const current = newMap.get(imagePath);
          if (current) {
            newMap.set(imagePath, {
              url: newUrl,
              loading: false,
              error: false,
              expiresAt: Date.now() + 3600000, // 1 hour from now
            });
          }
          return newMap;
        });
        // Schedule next refresh
        scheduleImageRefresh(imagePath, 3600);
      } catch (err) {
        console.error('Failed to refresh image URL:', err);
        setImageStates((prev) => {
          const newMap = new Map(prev);
          const current = newMap.get(imagePath);
          if (current) {
            newMap.set(imagePath, {
              ...current,
              error: true,
            });
          }
          return newMap;
        });
      }
    }, refreshDelay);

    refreshTimersRef.current.set(imagePath, timer);
  }, []);

  // Initialize image state when sources are added
  useEffect(() => {
    messages.forEach((message) => {
      if (message.sources) {
        message.sources.forEach((source) => {
          if (source.image_path && source.image_url) {
            const imageKey = source.image_path;
            setImageStates((prev) => {
              // Only initialize if not already present
              if (!prev.has(imageKey)) {
                const expiresAt = Date.now() + 3600000; // Assume 1 hour expiration
                const newState: ImageState = {
                  url: source.image_url!,
                  loading: true,
                  error: false,
                  expiresAt,
                };
                // Schedule refresh
                scheduleImageRefresh(imageKey, 3600);
                return new Map(prev).set(imageKey, newState);
              }
              return prev;
            });
          }
        });
      }
    });
  }, [messages, scheduleImageRefresh]);

  const handleImageLoad = (imagePath: string) => {
    setImageStates((prev) => {
      const newMap = new Map(prev);
      const current = newMap.get(imagePath);
      if (current) {
        newMap.set(imagePath, {
          ...current,
          loading: false,
          error: false,
        });
      }
      return newMap;
    });
  };

  const handleImageError = async (imagePath: string, source: SourceInfo) => {
    // Try to refresh the URL
    if (imagePath) {
      try {
        const newUrl = await api.getImageUrl(imagePath, 3600);
        setImageStates((prev) => {
          const newMap = new Map(prev);
          newMap.set(imagePath, {
            url: newUrl,
            loading: false,
            error: false,
            expiresAt: Date.now() + 3600000,
          });
          return newMap;
        });
        scheduleImageRefresh(imagePath, 3600);
      } catch (err) {
        console.error('Failed to refresh image URL on error:', err);
        setImageStates((prev) => {
          const newMap = new Map(prev);
          newMap.set(imagePath, {
            url: source.image_url || '',
            loading: false,
            error: true,
            expiresAt: Date.now() + 3600000,
          });
          return newMap;
        });
      }
    } else {
      setImageStates((prev) => {
        const newMap = new Map(prev);
        if (source.image_path) {
          newMap.set(source.image_path, {
            url: source.image_url || '',
            loading: false,
            error: true,
            expiresAt: Date.now() + 3600000,
          });
        }
        return newMap;
      });
    }
  };

  const handleImageClick = (url: string, alt: string) => {
    setExpandedImage({ url, alt });
  };

  const closeExpandedImage = () => {
    setExpandedImage(null);
  };

  // Handle ESC key to close lightbox
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && expandedImage) {
        closeExpandedImage();
      }
    };

    if (expandedImage) {
      window.addEventListener('keydown', handleKeyDown);
      // Prevent body scroll when lightbox is open
      document.body.style.overflow = 'hidden';
    }

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = '';
    };
  }, [expandedImage]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!input.trim() || isLoading) return;
    
    // Ensure at least one retriever is enabled
    if (!enableSparse && !enableDense && !enableTable && !enableImage && !enableGraph) {
      setError('Please select at least one retriever to use.');
      return;
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: input.trim(),
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setError(null);
    setIsLoading(true);

    // Add placeholder message for assistant response
    const assistantMessageId = (Date.now() + 1).toString();
    setMessages((prev) => [
      ...prev,
      {
        id: assistantMessageId,
        type: 'assistant',
        content: 'Thinking...',
        timestamp: new Date(),
      },
    ]);

    try {
      const response: QueryResponse = await api.query({
        query: userMessage.content,
        limit: 10,
        include_sources: true,
        enable_sparse: enableSparse,
        enable_dense: enableDense,
        enable_table: enableTable,
        enable_image: enableImage,
        enable_graph: enableGraph,
      });

      // Update assistant message with actual response
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMessageId
            ? {
                ...msg,
                content: response.answer,
                sources: response.sources,
              }
            : msg
        )
      );
    } catch (err: any) {
      const errorMessage =
        err.response?.data?.error ||
        err.response?.data?.detail?.error ||
        err.message ||
        'Failed to get response. Please try again.';
      
      setError(errorMessage);
      
      // Update assistant message with error
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMessageId
            ? {
                ...msg,
                content: `Sorry, I encountered an error: ${errorMessage}`,
              }
            : msg
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const toggleSources = (messageId: string) => {
    setExpandedSources((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(messageId)) {
        newSet.delete(messageId);
      } else {
        newSet.add(messageId);
      }
      return newSet;
    });
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h2>Chat with RAG Assistant</h2>
        <p className="chat-subtitle">
          Ask questions about your uploaded documents
        </p>
      </div>

      <div className="processor-config-wrapper">
        <div className="processor-config">
          <div className="processor-header">
            <h3>Retrievers</h3>
            <p className="processor-description">
              Select which retrievers to use for searching your documents
            </p>
          </div>
          <div className="processor-options">
            <label className={`processor-option ${enableSparse ? 'active' : ''}`}>
              <div className="processor-checkbox-wrapper">
                <input
                  type="checkbox"
                  checked={enableSparse}
                  onChange={(e) => setEnableSparse(e.target.checked)}
                  disabled={isLoading}
                />
                <span className="processor-label">BM25</span>
              </div>
              <span className="processor-hint">Sparse keyword search (BM25)</span>
            </label>
            <label className={`processor-option ${enableDense ? 'active' : ''}`}>
              <div className="processor-checkbox-wrapper">
                <input
                  type="checkbox"
                  checked={enableDense}
                  onChange={(e) => setEnableDense(e.target.checked)}
                  disabled={isLoading}
                />
                <span className="processor-label">Text</span>
              </div>
              <span className="processor-hint">Dense vector search for text</span>
            </label>
            <label className={`processor-option ${enableTable ? 'active' : ''}`}>
              <div className="processor-checkbox-wrapper">
                <input
                  type="checkbox"
                  checked={enableTable}
                  onChange={(e) => setEnableTable(e.target.checked)}
                  disabled={isLoading}
                />
                <span className="processor-label">Tables</span>
              </div>
              <span className="processor-hint">Dense vector search for tables</span>
            </label>
            <label className={`processor-option ${enableImage ? 'active' : ''}`}>
              <div className="processor-checkbox-wrapper">
                <input
                  type="checkbox"
                  checked={enableImage}
                  onChange={(e) => setEnableImage(e.target.checked)}
                  disabled={isLoading}
                />
                <span className="processor-label">Images</span>
              </div>
              <span className="processor-hint">Dense vector search for images</span>
            </label>
            <label className={`processor-option ${enableGraph ? 'active' : ''}`}>
              <div className="processor-checkbox-wrapper">
                <input
                  type="checkbox"
                  checked={enableGraph}
                  onChange={(e) => setEnableGraph(e.target.checked)}
                  disabled={isLoading}
                />
                <span className="processor-label">Knowledge Graph</span>
              </div>
              <span className="processor-hint">Graph-based retrieval using entity relationships</span>
            </label>
          </div>
        </div>
      </div>

      <div className="chat-messages" ref={messagesContainerRef}>
        {messages.map((message) => (
          <div
            key={message.id}
            className={`message ${message.type === 'user' ? 'user-message' : 'assistant-message'}`}
          >
            <div className="message-content">
              <div className="message-text">
                {message.type === 'assistant' ? (
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {message.content}
                  </ReactMarkdown>
                ) : (
                  message.content
                )}
              </div>
              
              {message.sources && message.sources.length > 0 && (
                <div className="message-sources">
                  <button
                    className="sources-toggle-button"
                    onClick={() => toggleSources(message.id)}
                    aria-expanded={expandedSources.has(message.id)}
                  >
                    <span className="sources-header">Sources ({message.sources.length})</span>
                    <svg
                      className={`sources-arrow ${expandedSources.has(message.id) ? 'expanded' : ''}`}
                      width="16"
                      height="16"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <polyline points="6 9 12 15 18 9"></polyline>
                    </svg>
                  </button>
                  {expandedSources.has(message.id) && (
                    <div className="sources-content">
                      {message.sources.map((source, index) => {
                    const hasImage = source.image_path && source.image_url;
                    const imageKey = source.image_path || '';
                    const imageState = imageKey ? imageStates.get(imageKey) : null;
                    const imageUrl = imageState?.url || source.image_url;
                    const isLoading = imageState?.loading ?? false;
                    const hasError = imageState?.error ?? false;

                    return (
                      <div key={source.chunk_id} className="source-item">
                        <span className="source-citation">
                          {source.citation}
                          {hasImage && (
                            <span className="source-image-badge" title="Contains image">
                              🖼️
                            </span>
                          )}
                        </span>
                        {hasImage && imageUrl && !hasError && (
                          <div className="source-image">
                            {isLoading && (
                              <div className="image-loading">
                                <div className="image-loading-spinner"></div>
                                <span>Loading image...</span>
                              </div>
                            )}
                            <img
                              src={imageUrl}
                              alt={source.chunk_text || source.citation || 'Source image'}
                              className="source-image-img"
                              onClick={() => handleImageClick(imageUrl, source.chunk_text || source.citation)}
                              onLoad={() => imageKey && handleImageLoad(imageKey)}
                              onError={() => handleImageError(imageKey, source)}
                              style={{ display: isLoading ? 'none' : 'block' }}
                            />
                          </div>
                        )}
                        {hasImage && hasError && (
                          <div className="image-error">
                            <span>⚠️ Image failed to load</span>
                            {imageKey && (
                              <button
                                className="image-retry-button"
                                onClick={() => handleImageError(imageKey, source)}
                              >
                                Retry
                              </button>
                            )}
                          </div>
                        )}
                        <div className="source-preview">{source.chunk_text}</div>
                      </div>
                    );
                      })}
                    </div>
                  )}
                </div>
              )}
              
              <div className="message-time">{formatTime(message.timestamp)}</div>
            </div>
          </div>
        ))}
        
        {isLoading && messages[messages.length - 1]?.type === 'assistant' && (
          <div className="message assistant-message">
            <div className="message-content">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}
      </div>

      {error && (
        <div className="error-message">
          <span>⚠️</span> {error}
          <button onClick={() => setError(null)} className="error-close">
            ×
          </button>
        </div>
      )}

      {/* Image Lightbox Modal */}
      {expandedImage && (
        <div className="image-lightbox-overlay" onClick={closeExpandedImage}>
          <div className="image-lightbox-content" onClick={(e) => e.stopPropagation()}>
            <button className="image-lightbox-close" onClick={closeExpandedImage} aria-label="Close">
              ×
            </button>
            <img
              src={expandedImage.url}
              alt={expandedImage.alt}
              className="image-lightbox-img"
            />
            <div className="image-lightbox-caption">{expandedImage.alt}</div>
          </div>
        </div>
      )}

      <form className="chat-input-form" onSubmit={handleSubmit}>
        <textarea
          ref={inputRef}
          className="chat-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about your documents..."
          rows={1}
          disabled={isLoading}
        />
        <button
          type="submit"
          className="chat-send-button"
          disabled={!input.trim() || isLoading}
        >
          {isLoading ? (
            <span className="spinner"></span>
          ) : (
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <line x1="22" y1="2" x2="11" y2="13"></line>
              <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
            </svg>
          )}
        </button>
      </form>
    </div>
  );
};

export default Chat;

