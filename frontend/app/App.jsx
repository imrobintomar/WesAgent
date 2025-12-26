import React, { useState, useMemo, useEffect, useRef } from "react";
import {
  Upload,
  Send,
  Loader,
  AlertCircle,
  CheckCircle,
  Download,
  Filter,
  X,
} from "lucide-react";
import VariantTable from "./components/VariantTable";
import VariantVisuals from "./components/VariantVisuals";
import "./App.css";

/* =========================
   API CONFIGURATION
 ========================= */

// Get API base URL from environment or use fallback
const getApiUrl = () => {
  // Priority: explicit env var > window location fallback
  const envUrl = import.meta.env.VITE_API_BASE_URL;
  
  if (envUrl) {
    console.log("✅ Using API URL from VITE_API_BASE_URL:", envUrl);
    return envUrl;
  }
  
  // Fallback: use window location
  console.warn(
    "⚠️ VITE_API_BASE_URL not set. Using window location.",
    "Set VITE_API_BASE_URL in .env for explicit API endpoint."
  );
  
  return window.location.origin;
};

const API_BASE = getApiUrl();

// Validate API URL
if (!API_BASE || API_BASE === "http://localhost:5173") {
  console.error(
    "❌ API_BASE is misconfigured!",
    "You must set VITE_API_BASE_URL in .env.local or .env.production"
  );
}

console.log("📍 API Base URL:", API_BASE);

/* =========================
   SAFE FIELD HELPERS
 ========================= */

function getGene(v) {
  return (
    v?.["Gene.refGeneWithVer"] ||
    v?.["Gene.refGene"] ||
    v?.gene ||
    ""
  );
}

function getAF(v) {
  const af =
    v?.af ??
    v?.AF ??
    v?.gnomAD_AF ??
    v?.gnomAD_exome_AF ??
    v?.gnomAD_genome_AF ??
    0;
  return Number(af) || 0;
}

/* =========================
   COMPONENT
 ========================= */

function App() {
  const [vcfFile, setVcfFile] = useState(null);
  const [variants, setVariants] = useState([]);
  const [analysis, setAnalysis] = useState("");
  const [workflowSteps, setWorkflowSteps] = useState([]);

  // Filters
  const [afFilter, setAfFilter] = useState(0.01);
  const [geneSearch, setGeneSearch] = useState("");

  const [userPrompt, setUserPrompt] = useState(
    "Identify all cancer-related variants and their therapeutic implications."
  );

  const [disease, setDisease] = useState("cancer");
  const [maxLitVariants, setMaxLitVariants] = useState(300);

  const [loading, setLoading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState("");
  const [hasResults, setHasResults] = useState(false);

  // Real-time progress
  const [progressMessages, setProgressMessages] = useState([]);
  const [showProgressPopup, setShowProgressPopup] = useState(false);
  const ws = useRef(null);

  // Cleanup WebSocket on unmount
  useEffect(() => {
    return () => {
      if (ws.current && ws.current.readyState === WebSocket.OPEN) {
        ws.current.close();
      }
    };
  }, []);

  /* =========================
     WEBSOCKET CONNECTION
  ========================= */

  const connectWebSocket = () => {
    // Close existing connection
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.close();
    }

    try {
      // Convert HTTP URL to WebSocket URL
      let wsUrl = API_BASE.replace(/^http:/, "ws:").replace(/^https:/, "wss:");
      wsUrl = wsUrl.replace(/\/$/, "") + "/ws/progress"; // Remove trailing slash and add endpoint

      console.log("🔌 Attempting WebSocket connection to:", wsUrl);

      ws.current = new WebSocket(wsUrl);

      ws.current.onopen = () => {
        console.log("✅ WebSocket Connected");
        setProgressMessages(["✅ Connected to analysis server..."]);
        setShowProgressPopup(true);
      };

      ws.current.onmessage = (event) => {
        console.log("📨 WebSocket message:", event.data);
        setProgressMessages((prev) => {
          const updated = [...prev, event.data];
          // Keep last 20 messages
          return updated.slice(-20);
        });
        setShowProgressPopup(true);
      };

      ws.current.onerror = (event) => {
        console.error("❌ WebSocket Error:", event);
        setProgressMessages((prev) => [
          ...prev,
          "⚠️ Connection issue (analysis may continue in background)",
        ]);
      };

      ws.current.onclose = () => {
        console.log("🔌 WebSocket Disconnected");
      };
    } catch (err) {
      console.error("Failed to create WebSocket:", err);
      setError("Could not connect to real-time updates (analysis will continue)");
    }
  };

  /* =========================
     FILTERED VARIANTS (SAFE)
  ========================= */

  const filteredVariants = useMemo(() => {
    if (!Array.isArray(variants) || variants.length === 0) return [];

    const q = geneSearch.toLowerCase();

    return variants.filter((v) => {
      const gene = getGene(v).toLowerCase();
      const af = getAF(v);

      const matchesGene = q === "" || gene.includes(q);
      const matchesAF = af <= afFilter;

      return matchesGene && matchesAF;
    });
  }, [variants, geneSearch, afFilter]);

  /* =========================
     FILE UPLOAD
  ========================= */

  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const validExtensions = [".vcf", ".vcf.gz", ".txt", ".tsv", ".csv"];
    const fileName = file.name.toLowerCase();
    const isValid = validExtensions.some((ext) => fileName.endsWith(ext));

    if (!isValid) {
      setError(`Please upload a supported file: ${validExtensions.join(", ")}`);
      return;
    }

    setError("");
    setVcfFile(file);
  };

  /* =========================
     ANALYSIS
  ========================= */

  const handleAnalyze = async () => {
    if (!vcfFile) {
      setError("Please upload a VCF file");
      return;
    }

    if (!userPrompt.trim()) {
      setError("Please enter an analysis prompt");
      return;
    }

    setLoading(true);
    setHasResults(false);
    setVariants([]);
    setAnalysis("");
    setWorkflowSteps([]);
    setError("");
    setProgressMessages([]);

    // Connect WebSocket for real-time updates
    connectWebSocket();

    // Create FormData object
    const formData = new FormData();

    console.log("🔧 Building FormData...");
    console.log(`  vcfFile: ${vcfFile.name} (${vcfFile.type}, ${vcfFile.size} bytes)`);
    console.log(`  prompt: ${userPrompt.substring(0, 50)}...`);
    console.log(`  disease: ${disease || "Unknown"}`);
    console.log(`  max_lit_variants: ${maxLitVariants || 300}`);

    formData.append("file", vcfFile, vcfFile.name);
    formData.append("prompt", userPrompt);
    formData.append("disease", disease || "Unknown");
    formData.append("max_lit_variants", String(maxLitVariants || 300));

    const analyzeUrl = `${API_BASE.replace(/\/$/, "")}/analyze`;
    console.log("🚀 Sending to:", analyzeUrl);

    try {
      setUploadProgress(0);

      const response = await fetch(analyzeUrl, {
        method: "POST",
        body: formData,
        credentials: "include", // Include cookies if needed for CORS
        // DO NOT set Content-Type header - let browser set it with boundary
        headers: {
          // Don't include Content-Type, browser will set it
          "Accept": "application/json",
        },
      });

      console.log(
        `📡 Response status: ${response.status} ${response.statusText}`
      );

      // Close WebSocket after delay
      setTimeout(() => {
        if (ws.current && ws.current.readyState === WebSocket.OPEN) {
          ws.current.close();
        }
      }, 2000);

      if (!response.ok) {
        let errorMsg = `HTTP ${response.status}: ${response.statusText}`;

        try {
          const errorData = await response.json();
          console.error("❌ Server error response:", errorData);

          if (errorData.detail) {
            if (Array.isArray(errorData.detail)) {
              errorMsg = errorData.detail
                .map((e) => `${e.loc?.join(".")} - ${e.msg}`)
                .join(", ");
            } else {
              errorMsg = errorData.detail;
            }
          } else if (errorData.error) {
            errorMsg = errorData.error;
          }
        } catch (parseErr) {
          console.warn("Could not parse error response:", parseErr);
        }

        setError(errorMsg);
        setLoading(false);

        // Log detailed error info for debugging CORS issues
        if (response.status === 503 || response.status === 0) {
          console.error(
            "⚠️ CORS or connectivity issue detected.",
            "Make sure VITE_API_BASE_URL is correct."
          );
          setError(
            errorMsg +
              "\n\nCORS/Connection Issue: Check that VITE_API_BASE_URL is set correctly."
          );
        }

        return;
      }

      const data = await response.json();
      console.log("✅ Analysis complete! Response:", data);

      setAnalysis(data?.results?.summary || "No summary returned.");

      const variantsData = Array.isArray(data?.results?.variants)
        ? data.results.variants
        : [];

      setVariants(variantsData);
      console.log(`📊 Loaded ${variantsData.length} variants`);

      setWorkflowSteps([
        `Input variants: ${data?.input_variants ?? "NA"}`,
        `Filtered variants: ${data?.filtered_variants ?? "NA"}`,
        `Prioritized for literature: ${data?.results?.prioritized_count ?? "NA"}`,
        `Literature evidence: ${data?.results?.literature_evidence_count ?? "NA"}`,
      ]);

      setHasResults(true);
      setProgressMessages((prev) => [...prev, "✅ Analysis complete!"]);
    } catch (err) {
      console.error("❌ Network/parsing error:", err);

      // Detailed error message for debugging
      let errorMsg = `Error: ${err.message}`;

      if (err.message.includes("Failed to fetch")) {
        errorMsg +=
          "\n\nPossible CORS issue or API endpoint unreachable.\nCheck:\n" +
          "1. VITE_API_BASE_URL is set correctly\n" +
          "2. Backend is running\n" +
          "3. No firewall/network blocking";
      }

      setError(errorMsg);
    } finally {
      setLoading(false);
      setUploadProgress(0);
    }
  };

  /* =========================
     EXPORT CSV
  ========================= */

  const exportToCSV = () => {
    if (filteredVariants.length === 0) {
      setError("No variants to export");
      return;
    }

    // Get all unique keys from variants
    const allKeys = new Set();
    filteredVariants.forEach((v) => {
      Object.keys(v).forEach((k) => allKeys.add(k));
    });

    const headers = Array.from(allKeys);
    const csvContent = [
      headers.join(","),
      ...filteredVariants.map((v) =>
        headers.map((h) => {
          const val = v[h];
          // Escape quotes and wrap in quotes if contains comma
          if (val === null || val === undefined) return "";
          const str = String(val);
          if (str.includes(",") || str.includes('"')) {
            return `"${str.replace(/"/g, '""')}"`;
          }
          return str;
        })
      ),
    ].join("\n");

    const blob = new Blob([csvContent], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `variants-${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  /* =========================
     UI
  ========================= */

  return (
    <div className="app">
      <header className="header">
        <h1>🧬 Whole Exome Sequencing Analysis Agent</h1>
        <p>Research-grade variant interpretation | AIIMS New Delhi</p>
      </header>

      <main className="main-content">
        <div className="grid">
          {/* LEFT PANEL */}
          <aside className="sidebar">
            <div className="panel">
              <h3>📁 Upload VCF</h3>

              <div
                className="upload-area"
                onClick={() => document.getElementById("vcf-upload")?.click()}
              >
                <Upload size={32} />
                <p>Click to upload VCF file</p>
              </div>

              <input
                id="vcf-upload"
                type="file"
                accept=".vcf,.vcf.gz,.txt,.tsv,.csv"
                hidden
                onChange={handleFileUpload}
              />

              {vcfFile && (
                <div className="success-badge">
                  <CheckCircle size={16} /> {vcfFile.name}
                </div>
              )}

              {loading && (
                <div className="progress-container">
                  <div className="progress-bar">
                    <div
                      className="progress-fill"
                      style={{ width: `${uploadProgress}%` }}
                    />
                  </div>
                  <small>Processing… {uploadProgress}%</small>
                </div>
              )}
            </div>

            <div className="panel">
              <h3>💬 Clinical Context</h3>

              <div className="filter-group">
                <label>Disease/Phenotype</label>
                <input
                  type="text"
                  className="input"
                  placeholder="e.g., Wilms Tumor, Lymphoma"
                  value={disease}
                  onChange={(e) => setDisease(e.target.value)}
                />
              </div>

              <div className="filter-group">
                <label>Max Variants for Literature ({maxLitVariants})</label>
                <input
                  type="number"
                  className="input"
                  min="5"
                  max="1000"
                  value={maxLitVariants}
                  onChange={(e) =>
                    setMaxLitVariants(Math.max(50, parseInt(e.target.value) || 300))
                  }
                />
                <small>Higher = slower but more thorough</small>
              </div>
            </div>

            <div className="panel">
              <h3>💬 Interpretation Prompt</h3>
              <textarea
                className="textarea"
                rows="5"
                value={userPrompt}
                onChange={(e) => setUserPrompt(e.target.value)}
              />

              <button
                className="btn btn-primary btn-full"
                onClick={handleAnalyze}
                disabled={loading || !vcfFile}
              >
                {loading ? (
                  <>
                    <Loader size={18} className="spin" /> Analyzing...
                  </>
                ) : (
                  <>
                    <Send size={18} /> Analyze Variants
                  </>
                )}
              </button>
            </div>

            {hasResults && (
              <div className="panel">
                <h3>
                  <Filter size={18} /> Interactive Filters
                </h3>

                <div className="filter-group">
                  <label>Search Gene</label>
                  <input
                    type="text"
                    className="input"
                    placeholder="e.g. TP53, KRAS"
                    value={geneSearch}
                    onChange={(e) => setGeneSearch(e.target.value)}
                  />
                </div>

                <div className="filter-group" style={{ marginTop: 10 }}>
                  <label>Max gnomAD AF: {afFilter.toFixed(4)}</label>
                  <input
                    type="range"
                    min="0"
                    max="0.10"
                    step="0.001"
                    value={afFilter}
                    onChange={(e) => setAfFilter(Number(e.target.value))}
                    style={{ width: "100%" }}
                  />
                </div>

                <button
                  className="btn btn-small"
                  onClick={exportToCSV}
                  style={{ marginTop: "10px", width: "100%" }}
                >
                  <Download size={16} /> Download CSV
                </button>
              </div>
            )}
          </aside>

          {/* RIGHT PANEL */}
          <section className="content">
            {error && (
              <div className="alert alert-error">
                <AlertCircle size={20} />
                <div style={{ marginLeft: "10px", whiteSpace: "pre-wrap" }}>
                  {error}
                </div>
              </div>
            )}

            {hasResults && <VariantVisuals variants={filteredVariants} />}

            {workflowSteps.length > 0 && (
              <div className="panel">
                <h3>🔄 Workflow</h3>
                <ul className="steps-list">
                  {workflowSteps.map((s, i) => (
                    <li key={i}>✓ {s}</li>
                  ))}
                </ul>
              </div>
            )}

            <div className="panel results-panel">
              <div className="results-header">
                <h3>📊 Analysis Results</h3>
              </div>

              {hasResults ? (
                <>
                  <pre className="report-text">{analysis}</pre>
                  <VariantTable
                    variants={filteredVariants}
                    geneQuery={geneSearch}
                  />
                </>
              ) : (
                <p className="empty-text">Upload a VCF file to begin analysis</p>
              )}
            </div>
          </section>
        </div>
      </main>

      {/* Progress Popup */}
      {showProgressPopup && (
        <div className="progress-popup">
          <div className="progress-popup-header">
            <h4>🔴 Live Analysis Progress</h4>
            <button
              onClick={() => setShowProgressPopup(false)}
              className="close-btn"
            >
              <X size={16} />
            </button>
          </div>
          <div className="progress-popup-content">
            {progressMessages.length === 0 ? (
              <div className="waiting-msg">
                <Loader size={14} className="spin" /> Initializing agents...
              </div>
            ) : (
              progressMessages.map((msg, i) => (
                <div key={i} className="progress-log-item">
                  <span className="log-bullet">»</span> {msg}
                </div>
              ))
            )}
          </div>
        </div>
      )}

      <footer className="footer">
        <p>For Research Purpose only | Contact Us</p>
      </footer>
    </div>
  );
}

export default App;