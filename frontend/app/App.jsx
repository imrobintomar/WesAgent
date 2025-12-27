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
    v?.["Gene.refGeneWithVer"] ||
    v?.["Gene.refGene"] ||
    v?.gene ||
    v?.["gene.refgene"] ||
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
          setVariants(Array.isArray(data?.variants) ? data.variants : []);
          setWorkflowSteps([
            `✓ Input: ${data?.variants_input || 0} variants`,
            `✓ Filtered: ${data?.variants_filtered || 0} variants`,
            `✓ Analysis complete`,
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
    if (!Array.isArray(variants) || variants.length === 0) return [];
    const q = geneSearch.toLowerCase();
    return variants.filter((v) => {
      const gene = getGene(v).toLowerCase();
      const af = getAF(v);
      return (q === "" || gene.includes(q)) && af <= afFilter;
    });
  }, [variants, geneSearch, afFilter]);

  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setVcfFile(file);
    setError(""); // Clear any previous errors
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
    setWorkflowSteps(["⏳ Submitting file..."]);
    setJobStatus("queued");
    connectWebSocket();

    const formData = new FormData();
    formData.append("file", vcfFile);
    formData.append("prompt", userPrompt);
    formData.append("disease", disease);
    formData.append("max_lit_variants", String(maxLitVariants));

    const baseUrl = API_BASE.replace(/\/$/, "");

    try {
      console.log("📤 Uploading file to:", baseUrl);
      const response = await fetch(`${baseUrl}/analyze`, {
        method: "POST",
        headers: {
          "ngrok-skip-browser-warning": "69420",
        },
        body: formData,
      });

      const data = await response.json();
      console.log("✅ Submit response:", data);

      if (!response.ok) {
        throw new Error(data.error || "Upload failed");
      }

      const jobId = data.job_id;
      console.log("🎯 Job ID:", jobId);

      setCurrentJobId(jobId);
      setProgressMessages((prev) => [
        ...prev,
        `✓ Job submitted: ${jobId}`,
      ].slice(-30));
      setWorkflowSteps([`✓ File submitted`, `⏳ Processing...`]);

      // Start polling for results
      startPolling(jobId);
    } catch (err) {
      console.error("❌ Upload error:", err);
      setError(err.message || "Failed to submit analysis");
      setLoading(false);
      setJobStatus("failed");
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
              <h3>📋 Analysis Settings</h3>
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
              <h3>💬 Analysis Prompt</h3>
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
                    <Loader className="spin" size={16} /> Analyzing...
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
                  AF Filter (≤ {afFilter.toFixed(3)})
                </label>
                <input
                  type="range"
                  min="0"
                  max="0.1"
                  step="0.001"
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
                <h3>🔄 Workflow</h3>
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

            {hasResults && filteredVariants.length > 0 && (
              <VariantVisuals variants={filteredVariants} />
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
                {filteredVariants.length > 0 && (
                  <>
                    <h4>Variants ({filteredVariants.length})</h4>
                    <VariantTable variants={filteredVariants} geneQuery={geneSearch} />
                  </>
                )}
                {filteredVariants.length === 0 && variants.length > 0 && (
                  <p className="no-variants">
                    No variants match current filters. Adjust AF threshold or gene search.
                  </p>
                )}
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