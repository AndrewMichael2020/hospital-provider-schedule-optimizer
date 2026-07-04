const fmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });

function card(label, value, note) {
  return `<article><span>${label}</span><strong>${value}</strong><small>${note}</small></article>`;
}

async function loadMetrics() {
  const response = await fetch("data/demo_case.json");
  if (!response.ok) throw new Error(`Unable to load metrics data: ${response.status}`);
  const data = await response.json();
  const summary = document.getElementById("metric-summary");
  const shortage = data.kpis.find((item) => item.label === "Shortage hours avoided");
  const risk = data.kpis.find((item) => item.label === "Coverage-risk hours avoided");
  const budget = data.kpis.find((item) => item.label === "Budget status");
  const remaining = data.remainingGaps.reduce((sum, item) => sum + Number(item.remainingGapHours || 0), 0);
  summary.innerHTML = [
    card(shortage.label, `${fmt.format(shortage.value)}h`, shortage.note),
    card(risk.label, `${fmt.format(risk.value)}h`, risk.note),
    card("Budget status", budget.value, budget.note),
    card("Remaining gaps", `${fmt.format(remaining)}h`, "30-day total uncovered role-hours"),
  ].join("");
  document.getElementById("data-source").textContent = `Scenario: ${data.scenario}; source: ${data.generatedFrom}`;
}

loadMetrics().catch((error) => {
  document.getElementById("metric-summary").innerHTML = card("Data unavailable", "Review", "Run the demo data builder and serve over HTTP.");
  console.error(error);
});
