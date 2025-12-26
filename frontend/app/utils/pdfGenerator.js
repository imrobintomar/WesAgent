import jsPDF from "jspdf";

export function generatePdf({ summary, variants }) {
  const pdf = new jsPDF();
  pdf.setFontSize(12);

  pdf.text("Cancer Variant Clinical Report", 10, 10);
  pdf.text("AIIMS New Delhi", 10, 18);

  pdf.setFontSize(10);
  pdf.text("Clinical Summary:", 10, 30);
  pdf.text(summary.slice(0, 1500), 10, 38);

  pdf.addPage();
  pdf.text("Key Variants:", 10, 10);

  variants.slice(0, 50).forEach((v, i) => {
    pdf.text(
      `${i + 1}. ${v.gene} | ${v.acmg.classification} | ${v.actionability.tier} | ${v.actionability.therapy || "N/A"}`,
      10,
      20 + i * 6
    );
  });

  pdf.save("clinical_report.pdf");
}
