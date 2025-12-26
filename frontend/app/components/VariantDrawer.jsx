import React from "react";
import { getGnomadAF, getCosmicCount } from "./variantUtils";

function VariantDrawer({ variant, onClose }) {
  if (!variant) return null;

  return (
    <div className="drawer right">
      <button onClick={onClose}>Close</button>
      <h3>Variant Annotation</h3>

      <p><b>gnomAD AF:</b> {getGnomadAF(variant) ?? "NA"}</p>
      <p><b>COSMIC:</b> {getCosmicCount(variant)}</p>

      <h4>ACMG</h4>
      <b>{variant.acmg?.classification || "NA"}</b>
      <ul>
        {(variant.acmg?.evidence || []).map(e => (
          <li key={e}>{e}</li>
        ))}
      </ul>

      <pre>{JSON.stringify(variant, null, 2)}</pre>
    </div>
  );
}

export default VariantDrawer;
