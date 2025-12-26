const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

export async function analyzeFile({ file, prompt, disease }) {
  const formData = new FormData();
  formData.append("file", file);        // MUST be "file"
  formData.append("prompt", prompt);
  formData.append("disease", disease);

  const res = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || "Analysis failed");
  }

  return res.json();
}
