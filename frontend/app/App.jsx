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
   SAFE FIELD HELPERS
 ========================= */

function getGene(v) {
  return (
    v?.["Gene.refGeneWithVer"] ||
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
  const [downloadCsv, setDownloadCsv] = useState(null);

  // Filters
  const [afFilter, setAfFilter] = useState(0.01);
  const [geneSearch, setGeneSearch] = useState("");

  const [userPrompt, setUserPrompt] = useState(
    "Identify all cancer-related variants and their therapeutic implications."
  );

  const [disease, setDisease] = useState("Unknown");
  const [maxLitVariants, setMaxLitVariants] = useState(300);

  const [loading, setLoading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState("");
  const [hasResults, setHasResults] = useState(false);

  // Real-time progress
  const [progressMessages, setProgressMessages] = useState([]);
  const [showProgressPopup, setShowProgressPopup] = useState(false);
  const ws = useRef(null);

  useEffect(() => {
    return () => {
      if (ws.current) {
        ws.current.close();
      }
    };
  }, []);

  const connectWebSocket = () => {
    if (ws.current) {
      ws.current.close();
    }

    try {
      // Use absolute URL from env if available, otherwise relative for proxy
      const apiBase = import.meta.env.VITE_API_BASE_URL;
      let wsUrl;

      if (apiBase && apiBase.startsWith("http")) {
        // Absolute URL for production/ngrok
        wsUrl = apiBase.replace(/^http/, "ws") + "/ws/progress";
      } else {
        // Relative URL for dev proxy
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const host = window.location.host;
        wsUrl = `${protocol}//${host}/ws/progress`;
      }

      console.log(`Attempting WebSocket connection to: ${wsUrl}`);

      ws.current = new WebSocket(wsUrl);

      ws.current.onmessage = (event) => {
        console.log("WebSocket message received:", event.data);
        setProgressMessages((prev) => [...prev, event.data].slice(-10));
        setShowProgressPopup(true);
      };

      ws.current.onopen = () => {
        console.log("✅ WebSocket Connected");
        setProgressMessages(["Connected to analysis server..."]);
        setShowProgressPopup(true);
      };

      ws.current.onerror = (event) => {
        console.error("❌ WebSocket Error:", event);
        setProgressMessages((prev) => [...prev, "⚠️ Connection issue (analysis may continue in background)"]);
      };

      ws.current.onclose = () => {
        console.log("WebSocket Disconnected");
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

      const matchesGene = gene.includes(q);
      const matchesAF = af <= afFilter;

      return matchesGene && matchesAF;
    });
  }, [variants, geneSearch, afFilter]);

  // Get API base URL dynamically
  const getApiBase = () => {
    // If VITE_API_BASE_URL is set (absolute URL), use it.
    // Otherwise, use empty string to allow Vite proxy to work in dev,
    // or fallback to origin if needed.
    return import.meta.env.VITE_API_BASE_URL || "";
  };

  const API_BASE = getApiBase();

  /* =========================
     FILE UPLOAD
  ========================= */

  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const validExtensions = [".vcf", ".vcf.gz", ".txt", ".tsv", ".csv"];
    const fileName = file.name.toLowerCase();
    const isValid = validExtensions.some(ext => fileName.endsWith(ext));

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
    setDownloadCsv(null);
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

    // Verify FormData contents
    console.log("FormData built. Sending to:", `${API_BASE}/analyze`);

    try {
      setUploadProgress(0);

      const response = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        body: formData,
        // Let browser set Content-Type with boundary automatically
      });

      console.log(`Response status: ${response.status} ${response.statusText}`);

      // Close WS after some delay
      setTimeout(() => {
        if (ws.current) {
          ws.current.close();
        }
      }, 2000);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        console.error(" Server error:", errorData);
        
        // Extract error message
        let errorMsg = response.statusText;
        if (errorData.detail) {
          if (Array.isArray(errorData.detail)) {
            errorMsg = errorData.detail.map(e => `${e.loc?.join('.')} - ${e.msg}`).join(", ");
          } else {
            errorMsg = errorData.detail;
          }
        }
        
        setError(`Error ${response.status}: ${errorMsg}`);
        setLoading(false);
        return;
      }

      const data = await response.json();
      console.log(" Analysis complete! Response:", data);

      setAnalysis(data?.results?.summary || "No summary returned.");

      const variantsData = Array.isArray(data?.results?.variants)
        ? data.results.variants
        : [];

      setVariants(variantsData);
      console.log(` Loaded ${variantsData.length} variants`);

      setWorkflowSteps([
        `Input variants: ${data?.input_variants ?? "NA"}`,
        `Filtered variants: ${data?.filtered_variants ?? "NA"}`,
        `Prioritized for literature: ${data?.results?.prioritized_count ?? "NA"}`,
        `Literature evidence: ${data?.results?.literature_evidence_count ?? "NA"}`,
        
      ]);

      setHasResults(true);
      setProgressMessages((prev) => [...prev, "Analysis complete!"]);
    } catch (err) {
      console.error(" Network/parsing error:", err);
      setError(`Error: ${err.message}`);
    } finally {
      setLoading(false);
      setUploadProgress(0);
    }
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
                  <small>Uploading… {uploadProgress}%</small>
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
                  onChange={(e) => setMaxLitVariants(Math.max(50, parseInt(e.target.value) || 300))}
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
                    <Loader size={18} className="spin" /> Analyzing Please Wait...
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
              </div>
            )}
          </aside>

          {/* RIGHT PANEL */}
          <section className="content">
            {error && (
              <div className="alert alert-error">
                <AlertCircle size={20} /> {error}
              </div>
            )}

            {hasResults && (
              <VariantVisuals variants={filteredVariants} />
            )}

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

                {hasResults && downloadCsv && (
                  <a href={downloadCsv} className="btn-small">
                    <Download size={16} /> Download Full CSV
                  </a>
                )}
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
                <p className="empty-text">
                  Upload a VCF file to begin analysis
                </p>
              )}
            </div>
          </section>
        </div>
      </main>

      {/* Progress Popup */}
      {showProgressPopup && (
        <div className="progress-popup">
          <div className="progress-popup-header">
            <h4>🔴 Live Agent Progress</h4>
            <button onClick={() => setShowProgressPopup(false)} className="close-btn">
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
        <p>For Research Purpose only | Contact Us </p>
      </footer>
    </div>
  );
}

export default App;