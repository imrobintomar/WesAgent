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

const getApiUrl = () => {
  const envUrl = import.meta.env.VITE_API_BASE_URL;
  if (envUrl) return envUrl;
  return "";
};

const API_BASE = getApiUrl();

/* =========================
   SAFE FIELD HELPERS
 ========================= */

function getGene(v) {
  return (
    v?.["Gene"] ||
    v?.["Gene.refGeneWithVer"] ||
    v?.["Gene.refGene"] ||
    v?.gene ||
    v?.["gene.refgene"] ||
    ""
  );
}

function getAF(v) {
  // Try multiple possible AF field names
  const af =
    v?.af ??
    v?.AF ??
    v?.VAF ??
    v?.gnomAD_AF ??
    v?.gnomAD_exome_AF ??
    v?.gnomAD_genome_AF ??
    v?.allele_freq ??
    0;
  const num = Number(af);
  return isNaN(num) ? 0 : num;
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
  const [afFilter, setAfFilter] = useState(1.0); // Show all variants by default
  const [geneSearch, setGeneSearch] = useState("");

  const [userPrompt, setUserPrompt] = useState(
    "Identify all cancer-related variants and their therapeutic implications."
  );

  const [disease, setDisease] = useState("cancer");
  const [maxLitVariants, setMaxLitVariants] = useState(300);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [hasResults, setHasResults] = useState(false);

  // Job tracking
  const [currentJobId, setCurrentJobId] = useState(null);
  const [jobStatus, setJobStatus] = useState(null); // queued, processing, completed, failed

  // Real-time progress
  const [progressMessages, setProgressMessages] = useState([]);
  const [showProgressPopup, setShowProgressPopup] = useState(false);
  const wsRef = useRef(null);
  const pollIntervalRef = useRef(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.close();
      }
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, []);

  const connectWebSocket = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.close();
    }

    try {
      let wsUrl;
      if (API_BASE.startsWith("http")) {
        wsUrl = API_BASE.replace(/^http/, "ws")
          .replace(/\/$/, "")
          .concat("/ws/progress");
      } else {
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        wsUrl = `${protocol}//${window.location.host}/ws/progress`;
      }

      console.log("🔌 Connecting WebSocket:", wsUrl);
      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = () => {
        console.log("✅ WebSocket connected");
        setProgressMessages(["✓ Connected to server..."]);
        setShowProgressPopup(true);
      };

      wsRef.current.onmessage = (event) => {
        console.log("📨 Progress:", event.data);
        setProgressMessages((prev) => [...prev, event.data].slice(-30));
      };

      wsRef.current.onerror = (err) => {
        console.warn("⚠️ WebSocket Error:", err);
        setProgressMessages((prev) => [
          ...prev,
          "⚠️ Connection issues - monitoring via polling...",
        ].slice(-30));
      };

      wsRef.current.onclose = () => {
        console.log("🔌 WebSocket disconnected");
      };
    } catch (err) {
      console.error("❌ WebSocket Error:", err);
      setProgressMessages((prev) => [
        ...prev,
        "❌ WebSocket failed - using polling only",
      ].slice(-30));
    }
  };

  // Start polling for job results
  const startPolling = (jobId) => {
    console.log("📊 Starting poll for job:", jobId);

    const pollResults = async () => {
      try {
        const baseUrl = API_BASE.replace(/\/$/, "");
        const res = await fetch(`${baseUrl}/results/${jobId}`, {
          headers: {
            "ngrok-skip-browser-warning": "69420",
          },
        });

        if (!res.ok) {
          console.error("❌ Polling response not OK:", res.status);
          return;
        }

        const contentType = res.headers.get("content-type");
        if (!contentType || !contentType.includes("application/json")) {
          const text = await res.text();
          console.error("❌ Expected JSON but received:", text.slice(0, 100));
          return;
        }

        const jobData = await res.json();
        console.log("📊 Job data:", jobData);

        setJobStatus(jobData.status);

        if (jobData.status === "processing") {
          setWorkflowSteps(["⏳ Parsing file...", "⏳ Analyzing variants..."]);
        }

        if (jobData.status === "completed") {
          console.log("✅ Analysis complete!");
          const data = jobData.results;

          setAnalysis(data?.summary || "");
          const variants = Array.isArray(data?.variants) ? data.variants : [];
          
          // Debug: Log first variant to check field names
          if (variants.length > 0) {
            console.log("📊 Sample variant:", variants[0]);
            console.log("📊 Total variants received:", variants.length);
            console.log("📊 Variant keys:", Object.keys(variants[0]));
          }
          
          setVariants(variants);
          
          // Use actual variants array length if metadata not available
          const inputCount = data?.variants_input || data?.input_variants || variants.length;
          const filteredCount = data?.variants_filtered || data?.filtered_variants || variants.length;
          
          setWorkflowSteps([
            `✓ Input: ${inputCount} variants`,
            `✓ Filtered: ${filteredCount} variants`,
            `✓ Analysis complete (${variants.length} cleaned records)`,
          ]);

          setHasResults(true);
          setLoading(false);

          // Stop polling
          if (pollIntervalRef.current) {
            clearInterval(pollIntervalRef.current);
            pollIntervalRef.current = null;
          }

          setProgressMessages((prev) => [
            ...prev,
            "✅ Analysis completed successfully!",
          ].slice(-30));
        } else if (
          jobData.status === "error" ||
          jobData.status === "failed"
        ) {
          console.error("❌ Job failed:", jobData.error);
          setError(jobData.error || "Analysis failed");
          setJobStatus("failed");
          setLoading(false);

          // Stop polling
          if (pollIntervalRef.current) {
            clearInterval(pollIntervalRef.current);
            pollIntervalRef.current = null;
          }

          setProgressMessages((prev) => [
            ...prev,
            `❌ Error: ${jobData.error}`,
          ].slice(-30));
        }
      } catch (pollErr) {
        console.error("❌ Polling error:", pollErr);
        setProgressMessages((prev) => [
          ...prev,
          `⚠️ Polling error: ${pollErr.message}`,
        ].slice(-30));
      }
    };

    // Poll every 2 seconds
    pollIntervalRef.current = setInterval(pollResults, 2000);

    // Also call it immediately
    pollResults();
  };

  const filteredVariants = useMemo(() => {
    if (!Array.isArray(variants) || variants.length === 0) {
      console.log("❌ No variants to filter");
      return [];
    }
    
    const q = geneSearch.toLowerCase();
    const filtered = variants.filter((v) => {
      const gene = getGene(v).toLowerCase();
      const af = getAF(v);
      const matches = (q === "" || gene.includes(q)) && af <= afFilter;
      return matches;
    });
    
    console.log(`📊 Filtering: ${filtered.length}/${variants.length} variants pass filter (query="${q}", AF≤${afFilter})`);
    return filtered;
  }, [variants, geneSearch, afFilter]);

  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setVcfFile(file);
    setError(""); // Clear any previous errors
  };

  // Check backend health before submitting
  const checkBackendHealth = async (baseUrl) => {
    try {
      console.log("🏥 Checking backend health...");
      const response = await fetch(`${baseUrl}/health`, {
        headers: {
          "ngrok-skip-browser-warning": "69420",
        },
      });

      if (!response.ok) {
        throw new Error(`Backend returned ${response.status}`);
      }

      const data = await response.json();
      console.log("✅ Backend healthy:", data);
      return true;
    } catch (err) {
      console.error("❌ Backend health check failed:", err);
      throw new Error(
        `Backend is not responding. Error: ${err.message}. Make sure the server is running.`
      );
    }
  };

  const handleAnalyze = async () => {
    if (!vcfFile) {
      setError("Please upload a file");
      return;
    }

    setLoading(true);
    setHasResults(false);
    setError("");
    setProgressMessages([]);
    setWorkflowSteps(["🏥 Checking backend...", "⏳ Submitting file..."]);
    setJobStatus("queued");
    connectWebSocket();

    const baseUrl = API_BASE.replace(/\/$/, "");

    try {
      // Check health first
      await checkBackendHealth(baseUrl);

      const formData = new FormData();
      formData.append("file", vcfFile);
      formData.append("prompt", userPrompt);
      formData.append("disease", disease);
      formData.append("max_lit_variants", String(maxLitVariants));

      console.log("📤 Uploading file to:", baseUrl);
      setWorkflowSteps(["✓ Backend OK", "📤 Uploading file..."]);

      const controller = new AbortController();
      // Increase timeout to 120 seconds for large files during read
      const timeoutId = setTimeout(() => controller.abort(), 120000);

      let response;
      try {
        response = await fetch(`${baseUrl}/analyze`, {
          method: "POST",
          headers: {
            "ngrok-skip-browser-warning": "69420",
          },
          body: formData,
          signal: controller.signal,
        });
      } finally {
        clearTimeout(timeoutId);
      }

      console.log("📨 Response status:", response.status);
      console.log("📨 Response headers:", Object.fromEntries(response.headers));

      let data;
      try {
        data = await response.json();
      } catch (jsonErr) {
        console.error("❌ Failed to parse JSON response:", jsonErr);
        const text = await response.text();
        console.error("Response text:", text);
        
        if (response.status === 524) {
          throw new Error(
            "Backend timeout (524). The analysis is queued but taking longer than expected. Try again in 30 seconds."
          );
        }
        
        throw new Error(
          `Server returned invalid JSON (${response.status}): ${text.slice(0, 100)}`
        );
      }

      console.log("✅ Submit response:", data);

      if (!response.ok) {
        throw new Error(data.error || `Server error: ${response.status}`);
      }

      const jobId = data.job_id;
      if (!jobId) {
        throw new Error("No job_id returned from server");
      }

      console.log("🎯 Job ID:", jobId);

      setCurrentJobId(jobId);
      setProgressMessages((prev) => [
        ...prev,
        `✓ Job submitted: ${jobId}`,
      ].slice(-30));
      setWorkflowSteps([`✓ Backend OK`, `✓ File uploaded`, `⏳ Processing...`]);

      // Start polling for results
      startPolling(jobId);
    } catch (err) {
      console.error("❌ Analysis error:", err);
      
      // Better error messages
      let errorMsg = err.message;
      if (err.name === "AbortError") {
        errorMsg = "Request timeout (>30s). Backend may be overloaded. Try again later.";
      } else if (errorMsg.includes("NetworkError")) {
        errorMsg = "Network error. Check your connection and make sure backend is running.";
      }
      
      setError(errorMsg);
      setLoading(false);
      setJobStatus("failed");
      setWorkflowSteps([]);
      setProgressMessages((prev) => [
        ...prev,
        `❌ Error: ${errorMsg}`,
      ].slice(-30));
    }
  };

  const exportToCSV = () => {
    if (filteredVariants.length === 0) {
      setError("No variants to export");
      return;
    }
    const headers = Object.keys(filteredVariants[0]);
    const csv = [
      headers.join(","),
      ...filteredVariants.map((v) =>
        headers.map((h) => JSON.stringify(v[h])).join(",")
      ),
    ].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "variants.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="app">
      <header className="header">
        <h1>🧬 Whole Exome Analysis Agent</h1>
        <p>Research-grade interpretation </p>
      </header>

      <main className="main-content">
        <div className="grid">
          <aside className="sidebar">
            <div className="panel">
              <h3>📁 Upload VCF</h3>
              <div
                className="upload-area"
                onClick={() => document.getElementById("vcf-upload")?.click()}
                style={{
                  opacity: loading ? 0.5 : 1,
                  cursor: loading ? "not-allowed" : "pointer",
                }}
              >
                <Upload size={32} />
                <p>{vcfFile ? vcfFile.name : "Click to upload"}</p>
              </div>
              <input
                id="vcf-upload"
                type="file"
                hidden
                onChange={handleFileUpload}
                disabled={loading}
              />
            </div>

            <div className="panel">
              <h3>Analysis Settings</h3>
              <label className="label-small">Disease/Phenotype</label>
              <input
                type="text"
                className="input"
                placeholder="e.g., Wilms, Cancer"
                value={disease}
                onChange={(e) => setDisease(e.target.value)}
                disabled={loading}
              />

              <label className="label-small" style={{ marginTop: 10 }}>
                Max Literature Variants
              </label>
              <input
                type="number"
                className="input"
                value={maxLitVariants}
                onChange={(e) => setMaxLitVariants(Number(e.target.value))}
                disabled={loading}
              />
            </div>

            <div className="panel">
              <h3> Analysis Prompt</h3>
              <textarea
                className="textarea"
                rows="5"
                value={userPrompt}
                onChange={(e) => setUserPrompt(e.target.value)}
                disabled={loading}
                placeholder="Describe what you want to analyze..."
              />
              <button
                className="btn btn-primary btn-full"
                onClick={handleAnalyze}
                disabled={loading || !vcfFile}
              >
                {loading ? (
                  <>
                    <Loader className="spin" size={16} /> Analyzing Please Wait...
                  </>
                ) : (
                  <>
                    <Send size={16} /> Analyze
                  </>
                )}
              </button>
            </div>

            {hasResults && (
              <div className="panel">
                <h3>
                  <Filter size={18} /> Filters
                </h3>
                <label className="label-small">Search Gene</label>
                <input
                  type="text"
                  className="input"
                  placeholder="e.g., TP53"
                  value={geneSearch}
                  onChange={(e) => setGeneSearch(e.target.value)}
                />

                <label className="label-small" style={{ marginTop: 10 }}>
                  AF Filter (≤ {afFilter >= 1 ? "All" : afFilter.toFixed(3)})
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={afFilter}
                  onChange={(e) => setAfFilter(Number(e.target.value))}
                />

                <button
                  className="btn btn-small"
                  onClick={exportToCSV}
                  style={{ marginTop: 15, width: "100%" }}
                >
                  <Download size={14} /> Download CSV
                </button>
              </div>
            )}
          </aside>

          <section className="content">
            {error && (
              <div className="alert alert-error">
                <AlertCircle size={18} /> {error}
              </div>
            )}

            {workflowSteps.length > 0 && (
              <div className="panel">
                <h3> Workflow</h3>
                <ul className="steps-list">
                  {workflowSteps.map((s, i) => (
                    <li key={i}>{s}</li>
                  ))}
                </ul>
              </div>
            )}

            {loading && currentJobId && (
              <div className="panel">
                <div className="status-info">
                  <Loader size={20} className="spin" />
                  <div>
                    <strong>Job in Progress</strong>
                    <p className="job-id-small">ID: {currentJobId}</p>
                  </div>
                </div>
              </div>
            )}

            {hasResults && variants.length > 0 && (
              <VariantVisuals variants={variants} />
            )}

            {hasResults && (
              <div className="panel results-panel">
                <h3>📊 Analysis Results</h3>
                {analysis && (
                  <>
                    <h4 style={{ marginTop: 0 }}>Summary</h4>
                    <pre className="report-text">{analysis}</pre>
                  </>
                )}
                {variants.length > 0 && (
                  <>
                    <h4>Variants ({variants.length})</h4>
                    <div style={{ marginBottom: "10px", fontSize: "0.9em", color: "#666" }}>
                      Showing {filteredVariants.length} of {variants.length} variants
                      {geneSearch && ` (gene: ${geneSearch})`}
                      {afFilter < 1 && ` (AF ≤ ${afFilter.toFixed(3)})`}
                    </div>
                    {filteredVariants.length > 0 ? (
                      <VariantTable variants={filteredVariants} geneQuery={geneSearch} />
                    ) : (
                      <p className="no-variants">
                        No variants match current filters. Try:
                        <ul style={{ marginLeft: "20px" }}>
                          <li>Increasing AF filter (drag slider right)</li>
                          <li>Clearing gene search</li>
                          <li>Resetting filters</li>
                        </ul>
                      </p>
                    )}
                  </>
                )}
                {variants.length === 0 && (
                  <p className="no-variants">No variants to display</p>
                )}
              </div>
            )}

            {!hasResults && !loading && error && (
              <div className="panel error-panel">
                <h3> Analysis Failed</h3>
                <p>{error}</p>
                <p style={{ fontSize: "0.9em", color: "#666", marginTop: "10px" }}>
                  <strong>Tips:</strong>
                </p>
                <ul style={{ fontSize: "0.9em", marginLeft: "20px" }}>
                  <li>Make sure file is valid VCF, CSV, or TSV format</li>
                  <li>Files can be gzip compressed (.vcf.gz, .txt.gz)</li>
                  <li>Check file encoding (UTF-8 preferred)</li>
                  <li>Try a smaller file first to test</li>
                </ul>
              </div>
            )}

            {!hasResults && !loading && !error && (
              <div className="panel empty-state">
                <p>📁 Upload a VCF file and configure your analysis</p>
              </div>
            )}
          </section>
        </div>
      </main>

      {showProgressPopup && (
        <div className="progress-popup">
          <div className="progress-popup-header">
            <div className="progress-header-title">
              <span
                className="pulse-dot"
                style={{
                  background:
                    jobStatus === "completed"
                      ? "#10b981"
                      : jobStatus === "failed"
                        ? "#ef4444"
                        : "#f59e0b",
                }}
              />
              <h4>Live Progress</h4>
            </div>
            <button
              onClick={() => setShowProgressPopup(false)}
              className="close-btn"
              title="Close"
            >
              <X size={16} />
            </button>
          </div>
          <div className="progress-popup-content">
            {progressMessages.length === 0 ? (
              <div className="progress-log-item">Waiting for updates...</div>
            ) : (
              progressMessages.map((msg, i) => (
                <div key={i} className="progress-log-item">
                  {msg}
                </div>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;