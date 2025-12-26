import React, { useEffect, useMemo, useRef } from "react";
import Plotly from "plotly.js-dist-min";

/* =========================
   LOW-LEVEL PLOT WRAPPER
========================= */

function Plot({ data, layout, height = 350 }) {
  const ref = useRef(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    Plotly.newPlot(
      el,
      data,
      { ...layout, height },
      {
        responsive: true,
        displayModeBar: false,
      }
    );

    return () => {
      Plotly.purge(el);
    };
  }, [data, layout, height]);

  return <div ref={ref} style={{ width: "100%" }} />;
}

/* =========================
   MAIN COMPONENT
========================= */

function VariantVisuals({ variants = [] }) {
  const stats = useMemo(() => {
    if (!Array.isArray(variants) || variants.length === 0) return null;

    const geneCounts = {};
    const effectCounts = {};
    const depths = [];
    const afs = [];
    const qualities = [];

    for (const v of variants) {
      const gene = v?.gene || "Unknown";
      const effect = v?.variant_effect || "Unknown";

      geneCounts[gene] = (geneCounts[gene] || 0) + 1;
      effectCounts[effect] = (effectCounts[effect] || 0) + 1;

      depths.push(Number(v?.depth || 0));
      afs.push(Number(v?.af || 0));
      qualities.push(Number(v?.quality || 0));
    }

    const topGenes = Object.keys(geneCounts)
      .sort((a, b) => geneCounts[b] - geneCounts[a])
      .slice(0, 10);

    return {
      topGenes,
      geneCounts,
      effectCounts,
      depths,
      afs,
      qualities,
    };
  }, [variants]);

  if (!stats) return null;

  const baseLayout = {
    paper_bgcolor: "#1f2937",
    plot_bgcolor: "#1f2937",
    font: { color: "#e5e7eb", size: 12 },
    margin: { t: 40, b: 50, l: 50, r: 20 },
  };

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(420px, 1fr))",
        gap: 20,
        marginBottom: 30,
      }}
    >
      <Plot
        data={[
          {
            x: stats.topGenes,
            y: stats.topGenes.map((g) => stats.geneCounts[g]),
            type: "bar",
            marker: { color: "#3b82f6" },
          },
        ]}
        layout={{
          ...baseLayout,
          title: "Top Mutated Genes",
          xaxis: { gridcolor: "#374151" },
          yaxis: { gridcolor: "#374151" },
        }}
      />

      <Plot
        data={[
          {
            labels: Object.keys(stats.effectCounts),
            values: Object.values(stats.effectCounts),
            type: "pie",
            hole: 0.4,
          },
        ]}
        layout={{
          ...baseLayout,
          title: "Variant Effects",
        }}
      />

      <Plot
        data={[
          {
            x: stats.depths,
            y: stats.afs,
            mode: "markers",
            type: "scatter",
            marker: { color: "#10b981", size: 6 },
          },
        ]}
        layout={{
          ...baseLayout,
          title: "AF vs Depth",
          xaxis: { title: "Depth", gridcolor: "#374151" },
          yaxis: { title: "Allele Frequency", gridcolor: "#374151" },
        }}
      />

      <Plot
        data={[
          {
            x: stats.qualities,
            type: "histogram",
            marker: { color: "#f59e0b" },
          },
        ]}
        layout={{
          ...baseLayout,
          title: "Quality Distribution",
          xaxis: { gridcolor: "#374151" },
          yaxis: { gridcolor: "#374151" },
        }}
      />
    </div>
  );
}

export default VariantVisuals;
