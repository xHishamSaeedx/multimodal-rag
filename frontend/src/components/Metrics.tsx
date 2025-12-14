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
        </div>

        {activeTab === "overview" && renderOverview()}
        {activeTab === "comparison" && renderComparison()}
      </div>
    </section>
  );
};

export default Metrics;
