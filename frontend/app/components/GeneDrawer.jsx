import React, { useMemo } from "react";
import GeneLollipopPlot from "./GeneLollipopPlot";
import { computeHotspots } from "./variantUtils";

function GeneDrawer({ gene, variants, onClose }) {
  const geneVars = useMemo(
    () =>
      variants.filter(
        v => (v["Gene.refGeneWithVer"] || v.gene) === gene
      ),
    [gene, variants]
  );

  const hotspots = computeHotspots(geneVars);

  if (!gene) return null;

  return (
    <div className="drawer left">
      <button onClick={onClose}>Close</button>
      <h3>Gene Summary: {gene}</h3>

      <p>Total Variants: {geneVars.length}</p>
      <p>Hotspot Clusters: {hotspots.length}</p>

      <GeneLollipopPlot gene={gene} variants={variants} />
    </div>
  );
}

export default GeneDrawer;
