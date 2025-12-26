import React, { useMemo } from "react";
import Plot from "react-plotly.js";
import {
  getProteinPosition,
  getCosmicCount,
  computeHotspots,
} from "./variantUtils";
import { PFAM_DOMAINS } from "./pfamData";

function GeneLollipopPlot({ gene, variants }) {
  const { points, hotspots, domains } = useMemo(() => {
    const geneVars = variants.filter(
      v => (v["Gene.refGeneWithVer"] || v.gene) === gene
    );

    const points = geneVars
      .map(v => {
        const pos = getProteinPosition(v);
        if (!pos) return null;
        return {
          pos,
          cosmic: getCosmicCount(v),
          label: v["AAChange.refGeneWithVer"],
        };
      })
      .filter(Boolean);

    return {
      points,
      hotspots: computeHotspots(geneVars),
      domains: PFAM_DOMAINS[gene] || [],
    };
  }, [gene, variants]);

  if (!gene || points.length === 0) return null;

  return (
    <Plot
      data={[
        // Lollipops
        {
          x: points.map(p => p.pos),
          y: points.map(() => 1),
          text: points.map(p => p.label),
          mode: "markers",
          marker: {
            size: points.map(p => Math.max(8, p.cosmic * 2)),
            color: points.map(p => p.cosmic),
            colorscale: "Reds",
          },
        },

        // Pfam domains
        ...domains.map(d => ({
          x: [d.start, d.end],
          y: [0.5, 0.5],
          mode: "lines",
          line: { width: 12 },
          name: d.name,
        })),
      ]}
      layout={{
        title: `Protein Lollipop – ${gene}`,
        height: 350,
        xaxis: { title: "Protein Position" },
        yaxis: { visible: false },
        shapes: hotspots.map(h => ({
          type: "rect",
          x0: h.center - 10,
          x1: h.center + 10,
          y0: 0,
          y1: 1.2,
          fillcolor: "rgba(239,68,68,0.2)",
          line: { width: 0 },
        })),
      }}
    />
  );
}

export default GeneLollipopPlot;
