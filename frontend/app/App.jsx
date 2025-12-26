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

  // Real-time progress
  const [progressMessages, setProgressMessages] = useState([]);
  const [showProgressPopup, setShowProgressPopup] = useState(false);
  const ws = useRef(null);

  useEffect(() => {
    return () => {
      if (ws.current && ws.current.readyState === WebSocket.OPEN) {
        ws.current.close();
      }
    };
  }, []);

  const connectWebSocket = () => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.close();
    }

    try {
      let wsUrl;
      if (API_BASE.startsWith("http")) {
        wsUrl = API_BASE.replace(/^http/, "ws").replace(/\/$/, "") + "/ws/progress";
      } else {
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        wsUrl = `${protocol}//${window.location.host}/ws/progress`;
      }

      console.log("🔌 Connecting WebSocket:", wsUrl);
      ws.current = new WebSocket(wsUrl);

      ws.current.onopen = () => {
        setProgressMessages(["✅ Connected to server..."]);
        setShowProgressPopup(true);
      };

      ws.current.onmessage = (event) => {
        setProgressMessages((prev) => [...prev, event.data].slice(-20));
        setShowProgressPopup(true);
      };

      ws.current.onerror = (err) => {
        console.warn("WebSocket Error:", err);
        setProgressMessages((prev) => [...prev, "⚠️ Connection issues. Falling back to polling..."].slice(-20));
      };
    } catch (err) {
      console.error("WS Error:", err);
    }
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
  };

  const handleAnalyze = async () => {
    if (!vcfFile) return setError("Please upload a file");
    
    setLoading(true);
    setHasResults(false);
    setError("");
    setProgressMessages([]);
    connectWebSocket();

    const formData = new FormData();
    formData.append("file", vcfFile);
    formData.append("prompt", userPrompt);
    formData.append("disease", disease);
    formData.append("max_lit_variants", String(maxLitVariants));

    const baseUrl = API_BASE.replace(/\/$/, "");
    
    try {
      const response = await fetch(`${baseUrl}/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error(await response.text());

      const { job_id } = await response.json();
      console.log("Job queued:", job_id);

      const poll = async () => {
        try {
          const res = await fetch(`${baseUrl}/results/${job_id}`);
          if (!res.ok) {
            console.error("Polling response not OK:", res.status);
            setTimeout(poll, 2000);
            return;
          }

          const contentType = res.headers.get("content-type");
          if (!contentType || !contentType.includes("application/json")) {
            const text = await res.text();
            console.error("Expected JSON but received:", text.slice(0, 100));
            setTimeout(poll, 2000);
            return;
          }

          const jobData = await res.json();

          if (jobData.status === "completed") {
            const data = jobData.results;
            setAnalysis(data?.summary || "");
            setVariants(data?.variants || []);
            setWorkflowSteps([
              `Input: ${data?.input_variants}`,
              `Filtered: ${data?.filtered_variants}`,
            ]);
            setHasResults(true);
            setLoading(false);
          } else if (jobData.status === "error" || jobData.status === "failed") {
            setError(jobData.error || "Analysis failed");
            setLoading(false);
          } else {
            setTimeout(poll, 2000);
          }
        } catch (pollErr) {
          console.error("Polling error:", pollErr);
          setTimeout(poll, 5000);
        }
      };
      poll();
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  const exportToCSV = () => {
    if (filteredVariants.length === 0) return;
    const headers = Object.keys(filteredVariants[0]);
    const csv = [headers.join(","), ...filteredVariants.map(v => headers.map(h => JSON.stringify(v[h])).join(","))].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "variants.csv";
    a.click();
  };

  return (
    <div className="app">
      <header className="header">
        <h1>🧬 WES Analysis Agent</h1>
        <p>Research-grade interpretation | AIIMS New Delhi</p>
      </header>

      <main className="main-content">
        <div className="grid">
          <aside className="sidebar">
            <div className="panel">
              <h3>📁 Upload VCF</h3>
              <div className="upload-area" onClick={() => document.getElementById("vcf-upload")?.click()}>
                <Upload size={32} />
                <p>{vcfFile ? vcfFile.name : "Click to upload"}</p>
              </div>
              <input id="vcf-upload" type="file" hidden onChange={handleFileUpload} />
            </div>

            <div className="panel">
              <h3>💬 Clinical Context</h3>
              <input type="text" className="input" placeholder="Disease" value={disease} onChange={e => setDisease(e.target.value)} />
              <input type="number" className="input" value={maxLitVariants} onChange={e => setMaxLitVariants(e.target.value)} />
            </div>

            <div className="panel">
              <h3>💬 Prompt</h3>
              <textarea className="textarea" rows="5" value={userPrompt} onChange={e => setUserPrompt(e.target.value)} />
              <button className="btn btn-primary btn-full" onClick={handleAnalyze} disabled={loading || !vcfFile}>
                {loading ? <Loader className="spin" /> : <Send />} Analyze
              </button>
            </div>

            {hasResults && (
              <div className="panel">
                <h3><Filter size={18} /> Filters</h3>
                <input type="text" className="input" placeholder="Search Gene" value={geneSearch} onChange={e => setGeneSearch(e.target.value)} />
                <input type="range" min="0" max="0.1" step="0.001" value={afFilter} onChange={e => setAfFilter(Number(e.target.value))} />
                <button className="btn btn-small" onClick={exportToCSV} style={{ marginTop: 10, width: "100%" }}>Download CSV</button>
              </div>
            )}
          </aside>

          <section className="content">
            {error && <div className="alert alert-error"><AlertCircle /> {error}</div>}
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
            {hasResults && <VariantVisuals variants={filteredVariants} />}
            {hasResults && (
              <div className="panel results-panel">
                <h3>📊 Results</h3>
                <pre className="report-text">{analysis}</pre>
                <VariantTable variants={filteredVariants} geneQuery={geneSearch} />
              </div>
            )}
          </section>
        </div>
      </main>

      {showProgressPopup && (
        <div className="progress-popup">
          <div className="progress-popup-header">
            <h4>🔴 Live Progress</h4>
            <button onClick={() => setShowProgressPopup(false)} className="close-btn"><X size={16} /></button>
          </div>
          <div className="progress-popup-content">
            {progressMessages.map((msg, i) => <div key={i} className="progress-log-item">» {msg}</div>)}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
