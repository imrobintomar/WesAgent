export function getField(v, keys) {
  for (const k of keys) {
    if (v[k] !== undefined && v[k] !== null && v[k] !== "") return v[k];
  }
  return null;
}

export function buildUniqueVariant(v) {
  const chr = getField(v, ["chrom", "Chr"]);
  const pos = getField(v, ["pos", "Start"]);
  const ref = getField(v, ["ref", "Ref"]);
  const alt = getField(v, ["alt", "Alt"]);
  return chr && pos && ref && alt ? `${chr}:${pos}::${alt}:${ref}` : "NA";
}

export function getProteinPosition(v) {
  const aa = v["AAChange.refGeneWithVer"];
  if (!aa) return null;
  const m = aa.match(/p\.[A-Z](\d+)[A-Z]/);
  return m ? Number(m[1]) : null;
}

export function getGnomadAF(v) {
  return getField(v, [
    "gnomAD_AF",
    "gnomAD_exome_AF",
    "gnomAD_genome_AF",
    "AF_popmax",
  ]);
}

export function getCosmicCount(v) {
  return getField(v, ["COSMIC_CNT", "cosmic70", "cosmic_coding"]) || 0;
}
