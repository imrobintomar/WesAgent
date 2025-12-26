import React, { useMemo, useState } from "react";

/* =========================
   UTILITIES
========================= */

function getField(v, keys) {
  for (const k of keys) {
    const val = v?.[k];
    if (val !== undefined && val !== null && val !== "") return val;
  }
  return "NA";
}

function getGene(v) {
  // Try multiple possible field names in order of preference
  return getField(v, [
    "Gene.refGeneWithVer",
    "Gene.refGene",
    "gene.refgenewithver",
    "gene.refgene",
    "gene",
    "Gene",
  ]);
}

function buildUniqueVariant(v) {
  const chr = getField(v, ["chrom", "Chr", "chromosome"]);
  const pos = getField(v, ["pos", "Start", "Position"]);
  const ref = getField(v, ["ref", "Ref"]);
  const alt = getField(v, ["alt", "Alt"]);
  if ([chr, pos, ref, alt].includes("NA")) return null;
  return `${chr}:${pos}:${ref}>${alt}`;
}

/* =========================
   VALIDATION
========================= */

function isValidVariant(v) {
  return (
    buildUniqueVariant(v) !== null &&
    Number(getField(v, ["DP", "depth"])) > 0 &&
    Number(getField(v, ["VAF", "vaf"])) > 0 &&
    getField(v, ["AAChange.refGeneWithVer", "AAChange.refGene", "AAChange"]) !== "NA"
  );
}

function isClinVarPathogenic(clnsig) {
  if (!clnsig || clnsig === "NA") return false;
  const v = String(clnsig).toLowerCase();
  return v === "pathogenic" || v === "likely_pathogenic";
}

/* =========================
   ACMG CONFIDENCE
========================= */

function computeACMGConfidence(v) {
  let score = 0;

  const exonic = getField(v, [
    "ExonicFunc.refGeneWithVer",
    "ExonicFunc.refGene",
    "ExonicFunc",
  ]);
  const clnsig = getField(v, ["CLNSIG"]);
  const vaf = Number(getField(v, ["VAF", "vaf"]));
  const dp = Number(getField(v, ["DP", "depth"]));

  if (["frameshift", "stopgain", "splicing"].includes(exonic)) score += 3;
  if (clnsig === "Pathogenic") score += 4;
  if (clnsig === "Likely_pathogenic") score += 3;
  if (dp >= 30) score += 1;
  if (vaf >= 0.1) score += 1;

  if (score >= 6) return { label: "High", color: "#16a34a" };
  if (score >= 4) return { label: "Moderate", color: "#f59e0b" };
  return { label: "Low", color: "#dc2626" };
}

/* =========================
   COMPONENT
========================= */

function VariantTable({ variants = [] }) {
  const [onlyPathogenic, setOnlyPathogenic] = useState(false);

  const processedVariants = useMemo(() => {
    let data = variants.filter(isValidVariant);

    if (onlyPathogenic) {
      data = data.filter((v) =>
        isClinVarPathogenic(getField(v, ["CLNSIG"]))
      );
    }

    return data;
  }, [variants, onlyPathogenic]);

  /* =========================
     EXPORT
  ========================= */

  const exportData = (delimiter) => {
    const headers = [
      "Variant",
      "Gene",
      "AAChange",
      "ExonicFunc",
      "DP",
      "VAF",
      "CLNSIG",
      "ACMG_Confidence",
    ];

    const rows = processedVariants.map((v) => {
      const acmg = computeACMGConfidence(v);
      return [
        buildUniqueVariant(v),
        getGene(v),
        getField(v, ["AAChange.refGeneWithVer", "AAChange.refGene", "AAChange"]),
        getField(v, [
          "ExonicFunc.refGeneWithVer",
          "ExonicFunc.refGene",
          "ExonicFunc",
        ]),
        getField(v, ["DP", "depth"]),
        getField(v, ["VAF", "vaf"]),
        getField(v, ["CLNSIG"]),
        acmg.label,
      ];
    });

    const content =
      headers.join(delimiter) +
      "\n" +
      rows.map((r) => r.join(delimiter)).join("\n");

    const blob = new Blob([content], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `variants.${delimiter === "," ? "csv" : "tsv"}`;
    a.click();
    URL.revokeObjectURL(url);
  };

  /* =========================
     UI
  ========================= */

  return (
    <div style={{ marginTop: 24 }}>
      <h3>Variant-Level Results (Validated)</h3>

      <div style={{ display: "flex", gap: 12, marginBottom: 8 }}>
        <button onClick={() => exportData(",")}>Export CSV</button>
        <button onClick={() => exportData("\t")}>Export TSV</button>

        <label>
          <input
            type="checkbox"
            checked={onlyPathogenic}
            onChange={(e) => setOnlyPathogenic(e.target.checked)}
          />{" "}
          ClinVar Pathogenic / Likely Pathogenic only
        </label>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns:
            "260px 160px 260px 140px 80px 80px 160px 120px",
          padding: 8,
          fontSize: 12,
          fontWeight: 600,
          background: "#1f2937",
          color: "#f9fafb",
        }}
      >
        <div>Variant</div>
        <div>Gene</div>
        <div>AAChange</div>
        <div>ExonicFunc</div>
        <div>DP</div>
        <div>VAF</div>
        <div>ClinVar</div>
        <div>ACMG</div>
      </div>

      <div style={{ height: 420, overflowY: "auto", background: "#111827" }}>
        {processedVariants.length === 0 ? (
          <div style={{ padding: 16, color: "#9ca3af", textAlign: "center" }}>
            No variants match the current filters
          </div>
        ) : (
          processedVariants.map((v, i) => {
            const acmg = computeACMGConfidence(v);
            return (
              <div
                key={i}
                style={{
                  display: "grid",
                  gridTemplateColumns:
                    "260px 160px 260px 140px 80px 80px 160px 120px",
                  padding: "4px 8px",
                  fontSize: 12,
                  borderBottom: "1px solid #374151",
                  color: "#e5e7eb",
                }}
              >
                <div>{buildUniqueVariant(v)}</div>
                <div style={{ fontWeight: 600, color: "#60a5fa" }}>
                  {getGene(v)}
                </div>
                <div>{getField(v, ["AAChange.refGeneWithVer", "AAChange.refGene", "AAChange"])}</div>
                <div>{getField(v, ["ExonicFunc.refGeneWithVer", "ExonicFunc.refGene", "ExonicFunc"])}</div>
                <div>{getField(v, ["DP", "depth"])}</div>
                <div>{getField(v, ["VAF", "vaf"])}</div>
                <div>{getField(v, ["CLNSIG"])}</div>
                <div
                  style={{
                    background: acmg.color,
                    color: "#000",
                    fontWeight: 700,
                    textAlign: "center",
                    borderRadius: 4,
                  }}
                >
                  {acmg.label}
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

export default VariantTable;