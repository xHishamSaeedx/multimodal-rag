import React, { useState, useEffect } from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { Bar, Line } from "react-chartjs-2";
import { api, ModelsConfig } from "../services/api";
import "./Metrics.css";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  Title,
  Tooltip,
  Legend
);

interface RetrieverMetrics {
  name: string;
  avgTime: number;
  relevanceScore: number;
  totalQueries: number;
  performanceRating: string;
  model?: string;
}

interface KnowledgeGraphMetrics {
  queryType: string;
  avgDuration: number;
  maxDuration: number;
  queriesExecuted: number;
}

const Metrics: React.FC = () => {
  const [activeTab, setActiveTab] = useState("overview");
  const [currentModels, setCurrentModels] = useState<ModelsConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Performance metrics mapping for different models
  const getModelMetrics = (
    modelName: string,
    retrieverType: string
  ): RetrieverMetrics | null => {
    // This mapping provides realistic performance data based on the performance reports
    const metricsMap: Record<string, RetrieverMetrics> = {
      // Sparse retriever (always Elasticsearch BM25)
      sparse: {
        name: "Sparse Retriever (BM25)",
        avgTime: 113.2,
        relevanceScore: 99.2,
        totalQueries: 6,
        performanceRating: "⭐⭐⭐ Excellent",
        model: "Elasticsearch BM25",
      },
      // Dense retrievers
      "Thenlper/GTE-Large": {
        name: "Dense Retriever (GTE-Large)",
        avgTime: 10.8,
        relevanceScore: 84.9,
        totalQueries: 5,
        performanceRating: "⭐⭐⭐⭐⭐ Exceptional",
        model: "Thenlper/GTE-Large (1024d)",
      },
      "intfloat/e5-large-v2": {
        name: "Dense Retriever (E5-Large)",
        avgTime: 10.5,
        relevanceScore: 80.3,
        totalQueries: 5,
        performanceRating: "⭐⭐⭐⭐⭐ Exceptional",
        model: "intfloat/e5-large-v2 (1024d)",
      },
      "Thenlper/GTE-Base": {
        name: "Dense Retriever (GTE-Base)",
        avgTime: 12.6,
        relevanceScore: 84.0,
        totalQueries: 5,
        performanceRating: "⭐⭐⭐⭐ Excellent",
        model: "Thenlper/GTE-Base (768d)",
      },
      "intfloat/e5-base-v2": {
        name: "Dense Retriever (E5-Base)",
        avgTime: 20.5,
        relevanceScore: 81.0,
        totalQueries: 6,
        performanceRating: "⭐⭐⭐ Good",
        model: "intfloat/e5-base-v2 (768d)",
      },
      "intfloat/multilingual-e5-large": {
        name: "Dense Retriever (Multilingual E5)",
        avgTime: 12.7,
        relevanceScore: 79.7,
        totalQueries: 5,
        performanceRating: "⭐⭐⭐⭐ Excellent",
        model: "intfloat/multilingual-e5-large (1024d)",
      },
      // Image retrievers
      "sentence-transformers/clip-ViT-L-14": {
        name: "Image Retriever (CLIP ViT-L-14)",
        avgTime: 51.2,
        relevanceScore: 27.4,
        totalQueries: 5,
        performanceRating: "⭐⭐ Moderate",
        model: "CLIP ViT-L-14 (768d)",
      },
      "CLIP ViT-B-32": {
        name: "Image Retriever (CLIP ViT-B-32)",
        avgTime: 505.1,
        relevanceScore: 15.8,
        totalQueries: 5,
        performanceRating: "🟡 Moderate-low",
        model: "CLIP ViT-B-32 (512d)",
      },
      SigLIP: {
        name: "Image Retriever (SigLIP)",
        avgTime: 509.2,
        relevanceScore: 1.9,
        totalQueries: 5,
        performanceRating: "🔴 Poor",
        model: "SigLIP vit_base_patch16_siglip_224 (768d)",
      },
    };

    return metricsMap[modelName] || metricsMap[retrieverType] || null;
  };

  // Get dynamic retriever metrics based on current models
  const getDynamicRetrieverMetrics = (): RetrieverMetrics[] => {
    if (!currentModels) return [];

    const metrics: RetrieverMetrics[] = [];

    // Always include sparse retriever
    const sparseMetrics = getModelMetrics("sparse", "sparse");
    if (sparseMetrics) metrics.push(sparseMetrics);

    // Add dense retriever with current model
    const denseMetrics = getModelMetrics(
      currentModels.models.text_embedding.model,
      "dense"
    );
    if (denseMetrics) metrics.push(denseMetrics);

    // Add image retriever with current model
    const imageModelKey =
      currentModels.models.image_embedding.model_name.includes("clip-ViT-L-14")
        ? "sentence-transformers/clip-ViT-L-14"
        : currentModels.models.image_embedding.model_name.includes("ViT-B-32")
        ? "CLIP ViT-B-32"
        : currentModels.models.image_embedding.model_name.includes("SigLIP")
        ? "SigLIP"
        : "sentence-transformers/clip-ViT-L-14";

    const imageMetrics = getModelMetrics(imageModelKey, "image");
    if (imageMetrics) metrics.push(imageMetrics);

    return metrics;
  };

  useEffect(() => {
    const fetchCurrentModels = async () => {
      try {
        setLoading(true);
        const modelsData = await api.getCurrentModels();
        setCurrentModels(modelsData);
        setError(null);
      } catch (err) {
        console.error("Failed to fetch current models:", err);
        setError("Failed to load current models configuration");
        // Fallback to default models if API fails
        setCurrentModels({
          status: "success",
          timestamp: new Date().toISOString(),
          models: {
            text_embedding: {
              model: "Thenlper/GTE-Large",
              dimension: 1024,
              device: "cuda",
            },
            image_embedding: {
              model_type: "clip",
              model_name: "sentence-transformers/clip-ViT-L-14",
            },
            llm: {
              provider: "openai",
              model: "gpt-4o",
            },
            captioning: {
              model: "Salesforce/blip-image-captioning-base",
            },
            vision_processing: {
              mode: "captioning",
            },
            retrieval_types: ["sparse", "dense", "image", "table"],
          },
        });
      } finally {
        setLoading(false);
      }
    };

    fetchCurrentModels();
  }, []);

  // Get retriever metrics dynamically
  const retrieverMetrics = getDynamicRetrieverMetrics();

  const modelComparison = [
    {
      model: "intfloat/e5-base-v2",
      denseTime: 20.5,
      denseRelevance: 81.0,
      dimensions: 768,
    },
    {
      model: "Thenlper/GTE-Base",
      denseTime: 12.6,
      denseRelevance: 84.0,
      dimensions: 768,
    },
    {
      model: "Thenlper/GTE-Large",
      denseTime: 10.8,
      denseRelevance: 84.9,
      dimensions: 1024,
    },
    {
      model: "intfloat/e5-large-v2",
      denseTime: 10.5,
      denseRelevance: 80.3,
      dimensions: 1024,
    },
    {
      model: "intfloat/multilingual-e5-large",
      denseTime: 12.7,
      denseRelevance: 79.7,
      dimensions: 1024,
    },
  ];

  const imageModelComparison = [
    {
      model: "CLIP ViT-L-14",
      imageTime: 51.2,
      imageRelevance: 27.4,
      dimensions: 768,
    },
    {
      model: "CLIP ViT-B-32",
      imageTime: 505.1,
      imageRelevance: 15.8,
      dimensions: 512,
    },
    {
      model: "SigLIP vit_base_patch16_siglip_224",
      imageTime: 509.2,
      imageRelevance: 1.9,
      dimensions: 768,
    },
  ];

  // Knowledge Graph Performance Data
  const knowledgeGraphMetrics: KnowledgeGraphMetrics[] = [
    {
      queryType: "graph_traversal",
      avgDuration: 107,
      maxDuration: 186,
      queriesExecuted: 5,
    },
    {
      queryType: "by_topics",
      avgDuration: 119,
      maxDuration: 214,
      queriesExecuted: 5,
    },
    {
      queryType: "by_section_title",
      avgDuration: 223,
      maxDuration: 422,
      queriesExecuted: 5,
    },
    {
      queryType: "by_keywords",
      avgDuration: 670,
      maxDuration: 1210,
      queriesExecuted: 5,
    },
  ];

  const knowledgeGraphOverview = {
    unifiedAverage: 243,
    maxRetrievalTime: 507,
    totalQueries: 20,
    totalChunksRetrieved: 94,
    avgChunksPerQuery: 18.7,
  };

  const renderOverview = () => {
    if (loading) {
      return (
        <div className="metrics-overview">
          <div className="metrics-header">
            <h2>Retriever Performance Overview</h2>
            <p>Loading current model configuration...</p>
          </div>
        </div>
      );
    }

    if (error || !currentModels) {
      return (
        <div className="metrics-overview">
          <div className="metrics-header">
            <h2>Retriever Performance Overview</h2>
            <p>Error loading model configuration. Using default values.</p>
          </div>
        </div>
      );
    }

    const performanceChartData = {
      labels: retrieverMetrics.map(
        (r) => r.name.split(" ")[0] + " " + r.name.split(" ")[1]
      ),
      datasets: [
        {
          label: "Relevance Score (%)",
          data: retrieverMetrics.map((r) => r.relevanceScore),
          backgroundColor: ["#007bff", "#28a745", "#ffc107"],
          borderColor: ["#0056b3", "#1e7e34", "#d39e00"],
          borderWidth: 1,
        },
      ],
    };

    const timeChartData = {
      labels: retrieverMetrics.map((r) => r.name.split(" ")[0]),
      datasets: [
        {
          label: "Response Time (ms)",
          data: retrieverMetrics.map((r) => r.avgTime),
          borderColor: "#000000",
          backgroundColor: "rgba(0, 0, 0, 0.1)",
          tension: 0.1,
        },
      ],
    };

    const chartOptions = {
      responsive: true,
      plugins: {
        legend: {
          position: "top" as const,
        },
      },
      scales: {
        y: {
          beginAtZero: true,
        },
      },
    };

    // Get the current dense model for insights
    const currentDenseModel = currentModels.models.text_embedding.model;
    const denseMetrics = getModelMetrics(currentDenseModel, "dense");
    const imageModelName =
      currentModels.models.image_embedding.model_name.includes("clip-ViT-L-14")
        ? "CLIP ViT-L-14"
        : currentModels.models.image_embedding.model_name.includes("ViT-B-32")
        ? "CLIP ViT-B-32"
        : "SigLIP";

    return (
      <div className="metrics-overview">
        <div className="metrics-header">
          <h2>Retriever Performance Overview</h2>
          <p>
            Comparative analysis of current retrieval methods in the multimodal
            RAG system
          </p>
        </div>

        <div className="metrics-charts">
          <div className="chart-container">
            <h3 className="chart-title">Relevance Scores by Retriever</h3>
            <Bar data={performanceChartData} options={chartOptions} />
          </div>

          <div className="chart-container">
            <h3 className="chart-title">Response Times Comparison</h3>
            <Line data={timeChartData} options={chartOptions} />
          </div>
        </div>

        <div className="metrics-grid">
          {retrieverMetrics.map((retriever, index) => (
            <div key={index} className="metric-card">
              <div className="metric-header">
                <h3>{retriever.name}</h3>
                <span className="model-info">{retriever.model}</span>
              </div>

              <div className="metric-stats">
                <div className="stat-item">
                  <span className="stat-label">Avg Response Time</span>
                  <span className="stat-value time">{retriever.avgTime}ms</span>
                </div>

                <div className="stat-item">
                  <span className="stat-label">Relevance Score</span>
                  <span className="stat-value relevance">
                    {retriever.relevanceScore}%
                  </span>
                </div>

                <div className="stat-item">
                  <span className="stat-label">Total Queries</span>
                  <span className="stat-value">{retriever.totalQueries}</span>
                </div>
              </div>

              <div className="performance-rating">
                <span className="rating-label">Performance:</span>
                <span className="rating-value">
                  {retriever.performanceRating}
                </span>
              </div>
            </div>
          ))}
        </div>

        <div className="metrics-summary">
          <h3>Key Insights</h3>
          <div className="insights-grid">
            <div className="insight-card">
              <h4>Speed Leader</h4>
              <p>
                Dense retriever with{" "}
                {currentDenseModel.split("/")[1] || currentDenseModel} achieves
                exceptional {denseMetrics?.avgTime || "N/A"}ms response time
              </p>
            </div>

            <div className="insight-card">
              <h4>Accuracy Champion</h4>
              <p>
                BM25 sparse search delivers near-perfect 99.2% relevance for
                keyword matching
              </p>
            </div>

            <div className="insight-card">
              <h4>Visual Processing</h4>
              <p>
                {imageModelName} vision model provides visual search
                capabilities
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderComparison = () => {
    const modelTimeData = {
      labels: modelComparison.map((m) => m.model.split("/")[1] || m.model),
      datasets: [
        {
          label: "Retrieval Time (ms)",
          data: modelComparison.map((m) => m.denseTime),
          backgroundColor: "rgba(40, 167, 69, 0.8)",
          borderColor: "#28a745",
          borderWidth: 1,
        },
      ],
    };

    const modelRelevanceData = {
      labels: modelComparison.map((m) => m.model.split("/")[1] || m.model),
      datasets: [
        {
          label: "Relevance Score (%)",
          data: modelComparison.map((m) => m.denseRelevance),
          backgroundColor: "rgba(0, 123, 255, 0.8)",
          borderColor: "#007bff",
          borderWidth: 1,
        },
      ],
    };

    const chartOptions = {
      responsive: true,
      plugins: {
        legend: {
          position: "top" as const,
        },
      },
      scales: {
        y: {
          beginAtZero: true,
        },
      },
    };

    return (
      <div className="metrics-comparison">
        <div className="metrics-header">
          <h2>Text Model Comparison</h2>
          <p>Performance evolution across five text embedding models tested</p>
        </div>

        <div className="comparison-charts">
          <div className="chart-container">
            <h3 className="chart-title">Text Model Response Times</h3>
            <Bar data={modelTimeData} options={chartOptions} />
          </div>

          <div className="chart-container">
            <h3 className="chart-title">Text Model Relevance Scores</h3>
            <Bar data={modelRelevanceData} options={chartOptions} />
          </div>
        </div>

        <div className="comparison-table">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Dense Retrieval Time</th>
                <th>Dense Relevance</th>
                <th>Dimensions</th>
                <th>Improvement</th>
              </tr>
            </thead>
            <tbody>
              {modelComparison.map((model, index) => (
                <tr key={index}>
                  <td className="model-name">{model.model}</td>
                  <td className="time-cell">{model.denseTime}ms</td>
                  <td className="relevance-cell">{model.denseRelevance}%</td>
                  <td className="dimensions-cell">{model.dimensions}</td>
                  <td className="improvement-cell">
                    {index === 0
                      ? "Baseline"
                      : index === 1
                      ? "⭐ +38% faster"
                      : index === 2
                      ? "⭐⭐ +47% faster"
                      : index === 3
                      ? "⭐⭐⭐ +49% faster"
                      : "⭐⭐⭐⭐ Multilingual"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="metrics-header">
          <h2>Vision Model Comparison</h2>
          <p>Performance comparison across CLIP and SigLIP vision models</p>
        </div>

        <div className="comparison-charts">
          <div className="chart-container">
            <h3 className="chart-title">Vision Model Response Times</h3>
            <Bar
              data={{
                labels: imageModelComparison.map(
                  (m) => m.model.split(" ")[1] || m.model
                ),
                datasets: [
                  {
                    label: "Retrieval Time (ms)",
                    data: imageModelComparison.map((m) => m.imageTime),
                    backgroundColor: "rgba(255, 193, 7, 0.8)",
                    borderColor: "#ffc107",
                    borderWidth: 1,
                  },
                ],
              }}
              options={chartOptions}
            />
          </div>

          <div className="chart-container">
            <h3 className="chart-title">Vision Model Relevance Scores</h3>
            <Bar
              data={{
                labels: imageModelComparison.map(
                  (m) => m.model.split(" ")[1] || m.model
                ),
                datasets: [
                  {
                    label: "Relevance Score (%)",
                    data: imageModelComparison.map((m) => m.imageRelevance),
                    backgroundColor: "rgba(220, 53, 69, 0.8)",
                    borderColor: "#dc3545",
                    borderWidth: 1,
                  },
                ],
              }}
              options={chartOptions}
            />
          </div>
        </div>

        <div className="comparison-table">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Image Retrieval Time</th>
                <th>Image Relevance</th>
                <th>Dimensions</th>
                <th>Performance</th>
              </tr>
            </thead>
            <tbody>
              {imageModelComparison.map((model, index) => (
                <tr key={index}>
                  <td className="model-name">{model.model}</td>
                  <td className="time-cell">{model.imageTime}ms</td>
                  <td className="relevance-cell">{model.imageRelevance}%</td>
                  <td className="dimensions-cell">{model.dimensions}</td>
                  <td className="improvement-cell">
                    {index === 0
                      ? "⭐⭐ Best Performance"
                      : index === 1
                      ? "🟡 Moderate"
                      : "🔴 Poor Performance"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="model-insights">
          <h3>Model Evolution Insights</h3>
          <div className="insights-list">
            <div className="insight-item">
              <span className="insight-icon">
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <line x1="12" y1="2" x2="12" y2="6"></line>
                  <line x1="12" y1="18" x2="12" y2="22"></line>
                  <line x1="4.93" y1="4.93" x2="7.76" y2="7.76"></line>
                  <line x1="16.24" y1="16.24" x2="19.07" y2="19.07"></line>
                  <line x1="2" y1="12" x2="6" y2="12"></line>
                  <line x1="18" y1="12" x2="22" y2="12"></line>
                  <line x1="4.93" y1="19.07" x2="7.76" y2="16.24"></line>
                  <line x1="16.24" y1="7.76" x2="19.07" y2="4.93"></line>
                </svg>
              </span>
              <div className="insight-content">
                <h4>Speed Evolution</h4>
                <p>
                  From 20.5ms (e5-base-v2) to 10.5ms (e5-large-v2) - 49%
                  performance improvement
                </p>
              </div>
            </div>

            <div className="insight-item">
              <span className="insight-icon">
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <circle cx="12" cy="12" r="10"></circle>
                  <circle cx="12" cy="12" r="6"></circle>
                  <circle cx="12" cy="12" r="2"></circle>
                </svg>
              </span>
              <div className="insight-content">
                <h4>Accuracy Trends</h4>
                <p>
                  GTE-Large achieves highest relevance (84.9%) while maintaining
                  excellent speed
                </p>
              </div>
            </div>

            <div className="insight-item">
              <span className="insight-icon">
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <circle cx="12" cy="12" r="10"></circle>
                  <path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path>
                </svg>
              </span>
              <div className="insight-content">
                <h4>Multilingual Support</h4>
                <p>
                  multilingual-e5-large provides cross-language capabilities
                  with 79.7% relevance
                </p>
              </div>
            </div>

            <div className="insight-item">
              <span className="insight-icon">
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                  <circle cx="9" cy="9" r="2"></circle>
                  <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"></path>
                </svg>
              </span>
              <div className="insight-content">
                <h4>Visual Model Performance</h4>
                <p>
                  CLIP ViT-L-14 outperforms SigLIP significantly (27.4% vs 1.9%
                  relevance) for visual search tasks
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderDocumentIngestion = () => {
    // Document ingestion performance metrics from the performance report
    const ingestionOverview = {
      totalTime: 25.605,
      fileSize: "318 KB",
      pages: 12,
      images: 5,
      tables: 5,
      textChunks: 10,
      totalChunks: 20,
      entities: 123,
      relationships: 3526,
    };

    const performanceBreakdown = [
      { category: "Storage Operations", time: 7.372, percentage: 28.8 },
      { category: "Table/Image Extraction", time: 6.64, percentage: 25.9 },
      { category: "Neo4j Graph Building", time: 5.583, percentage: 21.8 },
      { category: "Vision Processing", time: 2.857, percentage: 11.2 },
      { category: "Embedding Generation", time: 1.362, percentage: 5.3 },
      { category: "Elasticsearch Indexing", time: 0.292, percentage: 1.1 },
      { category: "Qdrant Vector Storage", time: 0.188, percentage: 0.7 },
      { category: "Text Processing", time: 0.004, percentage: 0.02 },
    ];

    const breakdownChartData = {
      labels: performanceBreakdown.map((item) => item.category),
      datasets: [
        {
          label: "Time (seconds)",
          data: performanceBreakdown.map((item) => item.time),
          backgroundColor: [
            "rgba(220, 53, 69, 0.8)",
            "rgba(255, 193, 7, 0.8)",
            "rgba(255, 152, 0, 0.8)",
            "rgba(156, 39, 176, 0.8)",
            "rgba(63, 81, 181, 0.8)",
            "rgba(76, 175, 80, 0.8)",
            "rgba(0, 188, 212, 0.8)",
            "rgba(96, 125, 139, 0.8)",
          ],
          borderColor: [
            "#dc3545",
            "#ffc107",
            "#ff9800",
            "#9c27b0",
            "#3f51b5",
            "#4caf50",
            "#00bcd4",
            "#607d8b",
          ],
          borderWidth: 1,
        },
      ],
    };

    const chartOptions = {
      responsive: true,
      plugins: {
        legend: {
          position: "top" as const,
        },
      },
      scales: {
        y: {
          beginAtZero: true,
        },
      },
    };

    return (
      <div className="metrics-document-ingestion">
        <div className="metrics-header">
          <h2>Document Ingestion Performance</h2>
          <p>
            Performance analysis of the multimodal document ingestion pipeline
            on a sample PDF document
          </p>
        </div>

        {/* Sample Document Info */}
        <div className="sample-document-banner">
          <div className="sample-info">
            <h3>📄 Sample Document: tech_sector_report.pdf</h3>
            <p>
              {ingestionOverview.fileSize} • {ingestionOverview.pages} pages •{" "}
              {ingestionOverview.images} images • {ingestionOverview.tables}{" "}
              tables
            </p>
          </div>
          <a
            href="http://localhost:8000/api/v1/health/sample-pdf"
            download="tech_sector_report.pdf"
            className="download-button"
          >
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="7 10 12 15 17 10"></polyline>
              <line x1="12" y1="15" x2="12" y2="3"></line>
            </svg>
            Download Sample PDF
          </a>
        </div>

        {/* Overview Cards */}
        <div className="ingestion-overview-cards">
          <div className="ingestion-stat-card">
            <div className="stat-icon">⏱️</div>
            <div className="stat-details">
              <h4>Total Ingestion Time</h4>
              <p className="stat-number">{ingestionOverview.totalTime}s</p>
              <span className="stat-description">End-to-end pipeline</span>
            </div>
          </div>

          <div className="ingestion-stat-card">
            <div className="stat-icon">📦</div>
            <div className="stat-details">
              <h4>Total Chunks Created</h4>
              <p className="stat-number">{ingestionOverview.totalChunks}</p>
              <span className="stat-description">
                {ingestionOverview.textChunks} text + {ingestionOverview.tables}{" "}
                table + {ingestionOverview.images} image
              </span>
            </div>
          </div>

          <div className="ingestion-stat-card">
            <div className="stat-icon">🔗</div>
            <div className="stat-details">
              <h4>Knowledge Graph</h4>
              <p className="stat-number">{ingestionOverview.entities}</p>
              <span className="stat-description">
                {ingestionOverview.relationships.toLocaleString()} relationships
              </span>
            </div>
          </div>

          <div className="ingestion-stat-card">
            <div className="stat-icon">✅</div>
            <div className="stat-details">
              <h4>Parallel Extraction</h4>
              <p className="stat-number">5.95s</p>
              <span className="stat-description">Time saved (23.2%)</span>
            </div>
          </div>
        </div>

        {/* Performance Breakdown Chart */}
        <div className="comparison-charts">
          <div className="chart-container">
            <h3 className="chart-title">Performance Breakdown by Category</h3>
            <Bar data={breakdownChartData} options={chartOptions} />
          </div>
        </div>

        {/* Detailed Breakdown Table */}
        <div className="comparison-table">
          <h3>Detailed Performance Breakdown</h3>
          <table>
            <thead>
              <tr>
                <th>Category</th>
                <th>Time (seconds)</th>
                <th>% of Total</th>
              </tr>
            </thead>
            <tbody>
              {performanceBreakdown.map((item, index) => (
                <tr key={index}>
                  <td className="model-name">{item.category}</td>
                  <td className="time-cell">{item.time}s</td>
                  <td className="relevance-cell">{item.percentage}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Granular Per-Item Metrics */}
        <div className="granular-metrics-section">
          <h3 className="section-title">Granular Per-Item Performance</h3>

          <div className="granular-metrics-grid">
            {/* Table Extraction Metrics */}
            <div className="granular-metric-card">
              <h4>📊 Table Extraction</h4>
              <div className="metric-detail">
                <span className="detail-label">Total Tables:</span>
                <span className="detail-value">5 tables</span>
              </div>
              <div className="metric-detail">
                <span className="detail-label">Total Time:</span>
                <span className="detail-value">6.640s</span>
              </div>
              <div className="metric-detail highlight">
                <span className="detail-label">Avg per Table:</span>
                <span className="detail-value">1.328s</span>
              </div>
              <div className="metric-detail">
                <span className="detail-label">Method:</span>
                <span className="detail-value">camelot-lattice</span>
              </div>
            </div>

            {/* Image Extraction Metrics */}
            <div className="granular-metric-card">
              <h4>🖼️ Image Extraction</h4>
              <div className="metric-detail">
                <span className="detail-label">Total Images:</span>
                <span className="detail-value">5 images</span>
              </div>
              <div className="metric-detail">
                <span className="detail-label">Total Time:</span>
                <span className="detail-value">5.950s</span>
              </div>
              <div className="metric-detail highlight">
                <span className="detail-label">Avg per Image:</span>
                <span className="detail-value">1.190s</span>
              </div>
              <div className="metric-detail">
                <span className="detail-label">Library:</span>
                <span className="detail-value">PyMuPDF (fitz)</span>
              </div>
            </div>

            {/* Vision Captioning Metrics */}
            <div className="granular-metric-card">
              <h4>🎯 Vision Captioning</h4>
              <div className="metric-detail">
                <span className="detail-label">Total Images:</span>
                <span className="detail-value">5 images</span>
              </div>
              <div className="metric-detail">
                <span className="detail-label">Total Time:</span>
                <span className="detail-value">2.857s</span>
              </div>
              <div className="metric-detail highlight">
                <span className="detail-label">Avg per Image:</span>
                <span className="detail-value">571ms</span>
              </div>
              <div className="metric-detail">
                <span className="detail-label">Model:</span>
                <span className="detail-value">BLIP-base</span>
              </div>
            </div>

            {/* Text Embeddings Metrics */}
            <div className="granular-metric-card">
              <h4>📝 Text Embeddings</h4>
              <div className="metric-detail">
                <span className="detail-label">Total Chunks:</span>
                <span className="detail-value">10 chunks</span>
              </div>
              <div className="metric-detail">
                <span className="detail-label">Total Time:</span>
                <span className="detail-value">0.806s</span>
              </div>
              <div className="metric-detail highlight">
                <span className="detail-label">Avg per Chunk:</span>
                <span className="detail-value">80.6ms</span>
              </div>
              <div className="metric-detail">
                <span className="detail-label">Model:</span>
                <span className="detail-value">GTE-Large (1024d)</span>
              </div>
            </div>

            {/* Table Embeddings Metrics */}
            <div className="granular-metric-card">
              <h4>📋 Table Embeddings</h4>
              <div className="metric-detail">
                <span className="detail-label">Total Tables:</span>
                <span className="detail-value">5 tables</span>
              </div>
              <div className="metric-detail">
                <span className="detail-label">Total Time:</span>
                <span className="detail-value">0.141s</span>
              </div>
              <div className="metric-detail highlight">
                <span className="detail-label">Avg per Table:</span>
                <span className="detail-value">28.2ms</span>
              </div>
              <div className="metric-detail">
                <span className="detail-label">Model:</span>
                <span className="detail-value">GTE-Large (1024d)</span>
              </div>
            </div>

            {/* Image Embeddings Metrics */}
            <div className="granular-metric-card">
              <h4>🖼️ Image Embeddings</h4>
              <div className="metric-detail">
                <span className="detail-label">Total Images:</span>
                <span className="detail-value">5 images</span>
              </div>
              <div className="metric-detail">
                <span className="detail-label">Total Time:</span>
                <span className="detail-value">0.415s</span>
              </div>
              <div className="metric-detail highlight">
                <span className="detail-label">Avg per Image:</span>
                <span className="detail-value">83ms</span>
              </div>
              <div className="metric-detail">
                <span className="detail-label">Model:</span>
                <span className="detail-value">CLIP (768d)</span>
              </div>
            </div>

            {/* Image Upload Metrics */}
            <div className="granular-metric-card">
              <h4>☁️ Image Uploads</h4>
              <div className="metric-detail">
                <span className="detail-label">Total Images:</span>
                <span className="detail-value">5 images</span>
              </div>
              <div className="metric-detail">
                <span className="detail-label">Total Time:</span>
                <span className="detail-value">3.441s</span>
              </div>
              <div className="metric-detail highlight">
                <span className="detail-label">Avg per Image:</span>
                <span className="detail-value">688ms</span>
              </div>
              <div className="metric-detail">
                <span className="detail-label">Storage:</span>
                <span className="detail-value">Supabase</span>
              </div>
            </div>

            {/* Database Operations Metrics */}
            <div className="granular-metric-card">
              <h4>💾 Database Inserts</h4>
              <div className="metric-detail">
                <span className="detail-label">Total Chunks:</span>
                <span className="detail-value">20 chunks</span>
              </div>
              <div className="metric-detail">
                <span className="detail-label">Total Time:</span>
                <span className="detail-value">3.865s</span>
              </div>
              <div className="metric-detail highlight">
                <span className="detail-label">Avg per Chunk:</span>
                <span className="detail-value">193ms</span>
              </div>
              <div className="metric-detail">
                <span className="detail-label">Database:</span>
                <span className="detail-value">Supabase</span>
              </div>
            </div>

            {/* Vector Storage Metrics */}
            <div className="granular-metric-card">
              <h4>🔍 Vector Storage</h4>
              <div className="metric-detail">
                <span className="detail-label">Total Vectors:</span>
                <span className="detail-value">20 vectors</span>
              </div>
              <div className="metric-detail">
                <span className="detail-label">Total Time:</span>
                <span className="detail-value">0.188s</span>
              </div>
              <div className="metric-detail highlight">
                <span className="detail-label">Avg per Vector:</span>
                <span className="detail-value">9.4ms</span>
              </div>
              <div className="metric-detail">
                <span className="detail-label">Database:</span>
                <span className="detail-value">Qdrant</span>
              </div>
            </div>
          </div>
        </div>

        {/* Key Insights */}
        <div className="model-insights">
          <h3>Performance Insights</h3>
          <div className="insights-list">
            <div className="insight-item">
              <span className="insight-icon">
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
                </svg>
              </span>
              <div className="insight-content">
                <h4>🚀 Excellent Performance</h4>
                <p>
                  Parallel extraction working effectively - saved 5.95 seconds
                  (23.2%). Fast vector storage with Qdrant (0.188s) and
                  efficient text processing (0.004s).
                </p>
              </div>
            </div>

            <div className="insight-item">
              <span className="insight-icon">
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <circle cx="12" cy="12" r="10"></circle>
                  <path d="M12 6v6l4 2"></path>
                </svg>
              </span>
              <div className="insight-content">
                <h4>🔴 Primary Bottlenecks</h4>
                <p>
                  Storage operations (28.8%), table/image extraction (25.9%),
                  and Neo4j graph building (21.8%) account for 76.5% of total
                  time. Focus optimization efforts here.
                </p>
              </div>
            </div>

            <div className="insight-item">
              <span className="insight-icon">
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"></path>
                </svg>
              </span>
              <div className="insight-content">
                <h4>🎯 Optimization Potential</h4>
                <p>
                  Conservative optimizations could reduce time to 22.6s (11.7%
                  faster). Aggressive optimizations with optional graph building
                  could reach 13.0s (49.2% faster).
                </p>
              </div>
            </div>

            <div className="insight-item">
              <span className="insight-icon">
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path>
                </svg>
              </span>
              <div className="insight-content">
                <h4>💡 Key Recommendations</h4>
                <p>
                  Optimize Supabase image uploads (parallel uploads,
                  compression), make Neo4j graph building optional, and profile
                  table extraction for library-level optimizations.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderKnowledgeGraph = () => {
    const queryTypeTimeData = {
      labels: knowledgeGraphMetrics.map((m) =>
        m.queryType.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase())
      ),
      datasets: [
        {
          label: "Average Duration (ms)",
          data: knowledgeGraphMetrics.map((m) => m.avgDuration),
          backgroundColor: "rgba(75, 192, 192, 0.8)",
          borderColor: "#4bc0c0",
          borderWidth: 1,
        },
        {
          label: "Max Duration (ms)",
          data: knowledgeGraphMetrics.map((m) => m.maxDuration),
          backgroundColor: "rgba(255, 99, 132, 0.8)",
          borderColor: "#ff6384",
          borderWidth: 1,
        },
      ],
    };

    const chartOptions = {
      responsive: true,
      plugins: {
        legend: {
          position: "top" as const,
        },
      },
      scales: {
        y: {
          beginAtZero: true,
        },
      },
    };

    return (
      <div className="metrics-knowledge-graph">
        <div className="metrics-header">
          <h2>Knowledge Graph Performance</h2>
          <p>
            Performance analysis of Neo4j graph-based retrieval across 4 query
            types
          </p>
        </div>

        {/* Overview Cards */}
        <div className="graph-overview-cards">
          <div className="graph-stat-card">
            <div className="stat-icon">📊</div>
            <div className="stat-details">
              <h4>Unified Average</h4>
              <p className="stat-number">
                {knowledgeGraphOverview.unifiedAverage}ms
              </p>
              <span className="stat-description">Average retrieval time</span>
            </div>
          </div>

          <div className="graph-stat-card">
            <div className="stat-icon">⚡</div>
            <div className="stat-details">
              <h4>Max Retrieval</h4>
              <p className="stat-number">
                {knowledgeGraphOverview.maxRetrievalTime}ms
              </p>
              <span className="stat-description">Peak latency observed</span>
            </div>
          </div>

          <div className="graph-stat-card">
            <div className="stat-icon">🔍</div>
            <div className="stat-details">
              <h4>Total Queries</h4>
              <p className="stat-number">
                {knowledgeGraphOverview.totalQueries}
              </p>
              <span className="stat-description">Across all query types</span>
            </div>
          </div>

          <div className="graph-stat-card">
            <div className="stat-icon">📦</div>
            <div className="stat-details">
              <h4>Chunks Retrieved</h4>
              <p className="stat-number">
                {knowledgeGraphOverview.avgChunksPerQuery}
              </p>
              <span className="stat-description">Average per query</span>
            </div>
          </div>
        </div>

        {/* Performance Charts */}
        <div className="comparison-charts">
          <div className="chart-container">
            <h3 className="chart-title">Query Type Performance Comparison</h3>
            <Bar data={queryTypeTimeData} options={chartOptions} />
          </div>
        </div>

        {/* Detailed Table */}
        <div className="comparison-table">
          <h3>Query Type Performance Details</h3>
          <table>
            <thead>
              <tr>
                <th>Query Type</th>
                <th>Average Duration</th>
                <th>Max Duration</th>
                <th>Queries Executed</th>
              </tr>
            </thead>
            <tbody>
              {knowledgeGraphMetrics.map((metric, index) => (
                <tr key={index}>
                  <td className="model-name">
                    {metric.queryType
                      .replace(/_/g, " ")
                      .replace(/\b\w/g, (l) => l.toUpperCase())}
                  </td>
                  <td className="time-cell">{metric.avgDuration}ms</td>
                  <td className="time-cell">
                    {metric.maxDuration > 1000
                      ? `${(metric.maxDuration / 1000).toFixed(2)}s`
                      : `${metric.maxDuration}ms`}
                  </td>
                  <td>{metric.queriesExecuted}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Key Insights */}
        <div className="model-insights">
          <h3>Knowledge Graph Insights</h3>
          <div className="insights-list">
            <div className="insight-item">
              <span className="insight-icon">
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <circle cx="12" cy="12" r="10"></circle>
                  <path d="M8 14s1.5 2 4 2 4-2 4-2"></path>
                  <line x1="9" y1="9" x2="9.01" y2="9"></line>
                  <line x1="15" y1="9" x2="15.01" y2="9"></line>
                </svg>
              </span>
              <div className="insight-content">
                <h4>Fastest Query Type</h4>
                <p>
                  Graph traversal queries achieve the best performance at 107ms
                  average, making them ideal for relationship-based searches
                </p>
              </div>
            </div>

            <div className="insight-item">
              <span className="insight-icon">
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
                </svg>
              </span>
              <div className="insight-content">
                <h4>Performance Spread</h4>
                <p>
                  6.3x performance difference between fastest (graph_traversal:
                  107ms) and slowest (by_keywords: 670ms) query types
                </p>
              </div>
            </div>

            <div className="insight-item">
              <span className="insight-icon">
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
                  <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
                  <line x1="12" y1="22.08" x2="12" y2="12"></line>
                </svg>
              </span>
              <div className="insight-content">
                <h4>High Retrieval Volume</h4>
                <p>
                  Knowledge graph retrieves 18.7 chunks per query on average,
                  87% more than hybrid retrieval (10.0 chunks)
                </p>
              </div>
            </div>

            <div className="insight-item">
              <span className="insight-icon">
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"></path>
                </svg>
              </span>
              <div className="insight-content">
                <h4>Optimization Opportunity</h4>
                <p>
                  By_keywords queries (670ms avg) show potential for
                  optimization through query rewriting or caching strategies
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <section id="metrics" className="metrics">
      <div className="metrics-container">
        <div className="metrics-navigation">
          <button
            className={`nav-tab ${activeTab === "overview" ? "active" : ""}`}
            onClick={() => setActiveTab("overview")}
          >
            Overview
          </button>
          <button
            className={`nav-tab ${activeTab === "comparison" ? "active" : ""}`}
            onClick={() => setActiveTab("comparison")}
          >
            Model Comparison
          </button>
          <button
            className={`nav-tab ${
              activeTab === "knowledge-graph" ? "active" : ""
            }`}
            onClick={() => setActiveTab("knowledge-graph")}
          >
            Knowledge Graph
          </button>
          <button
            className={`nav-tab ${
              activeTab === "document-ingestion" ? "active" : ""
            }`}
            onClick={() => setActiveTab("document-ingestion")}
          >
            Document Ingestion
          </button>
        </div>

        {activeTab === "overview" && renderOverview()}
        {activeTab === "comparison" && renderComparison()}
        {activeTab === "knowledge-graph" && renderKnowledgeGraph()}
        {activeTab === "document-ingestion" && renderDocumentIngestion()}
      </div>
    </section>
  );
};

export default Metrics;
