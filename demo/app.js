const fmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });
const pct = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1, style: "percent" });

function byId(id) {
  return document.getElementById(id);
}

function maxOf(items, fields) {
  return Math.max(1, ...items.flatMap((item) => fields.map((field) => Number(item[field]) || 0)));
}

function renderKpis(kpis) {
  byId("kpi-grid").innerHTML = kpis
    .map((kpi, index) => {
      const isStatus = typeof kpi.value === "string";
      return `
        <article class="kpi-card ${index === 0 ? "kpi-card--primary" : ""}">
          <p>${kpi.label}</p>
          <strong class="${isStatus ? "status-value" : ""}">${isStatus ? kpi.value : fmt.format(kpi.value)}${kpi.unit ? `<span>${kpi.unit}</span>` : ""}</strong>
          <small>${kpi.note}</small>
        </article>
      `;
    })
    .join("");
}

function renderComparison(items) {
  const max = maxOf(items, ["shortageHours", "coverageRiskHours"]);
  byId("comparison-chart").innerHTML = items
    .map((item) => {
      const shortagePct = (Number(item.shortageHours) / max) * 100;
      const riskPct = (Number(item.coverageRiskHours) / max) * 100;
      return `
        <div class="bar-row">
          <div class="bar-row__label">${item.mode}</div>
          <div class="bar-stack">
            <div class="bar-line">
              <span>Shortage</span>
              <div class="bar-track"><b style="width:${shortagePct}%"></b></div>
              <strong>${fmt.format(item.shortageHours)}</strong>
            </div>
            <div class="bar-line bar-line--risk">
              <span>Risk</span>
              <div class="bar-track"><b style="width:${riskPct}%"></b></div>
              <strong>${fmt.format(item.coverageRiskHours)}</strong>
            </div>
          </div>
        </div>
      `;
    })
    .join("");
}

function renderBudgets(items) {
  byId("budget-chart").innerHTML = items
    .map(
      (item) => `
        <div class="meter">
          <div class="meter__top">
            <strong>${item.role}</strong>
            <span>${fmt.format(item.usedHours)} / ${fmt.format(item.budgetHours)} hours</span>
          </div>
          <div class="meter__track">
            <b class="${item.overBudget ? "is-over" : ""}" style="width:${Math.min(100, item.usedPct)}%"></b>
          </div>
          <small>${fmt.format(item.remainingHours)} hours remaining, ${fmt.format(item.usedPct)}% used</small>
        </div>
      `,
    )
    .join("");
}

function renderActions(items) {
  byId("actions-table").innerHTML = items
    .map(
      (item) => `
        <tr>
          <td data-label="Facility">${item.facility}</td>
          <td data-label="Role">${item.role}</td>
          <td data-label="Window">${item.window}</td>
          <td data-label="Priority"><span class="pill">${item.priority}</span></td>
          <td data-label="Expected hit">${item.expectedHit ? "Yes" : "Review"}</td>
        </tr>
      `,
    )
    .join("");
}

function renderGaps(items) {
  const max = maxOf(items, ["remainingGapHours"]);
  byId("gaps-chart").innerHTML = items
    .map((item) => {
      const pct = (Number(item.remainingGapHours) / max) * 100;
      return `
        <div class="gap-row">
          <strong>${item.role}</strong>
          <div class="bar-track"><b style="width:${pct}%"></b></div>
          <span>${fmt.format(item.remainingGapHours)} gap hours</span>
          <small>${item.pressureHours} pressure hours</small>
        </div>
      `;
    })
    .join("");
}

async function loadDemo() {
  const response = await fetch("data/demo_case.json");
  if (!response.ok) {
    throw new Error(`Unable to load demo data: ${response.status}`);
  }
  const data = await response.json();
  byId("case-subtitle").textContent = data.case.subtitle;
  byId("case-network").textContent = `${data.case.facilities} facilities / ${data.case.days} days`;
  byId("case-scenario").textContent = data.scenario.replaceAll("_", " ");
  byId("story-problem").textContent = data.plainLanguage.problem;
  byId("story-manual").textContent = data.plainLanguage.manual;
  byId("story-optimizer").textContent = data.plainLanguage.optimizer;
  byId("story-remaining").textContent = data.plainLanguage.remaining;
  byId("data-source").textContent = `Scenario: ${data.scenario}. Generated from: ${data.generatedFrom}`;
  const overBudget = data.roleBudgets.some((item) => item.overBudget);
  const totalUsed = data.roleBudgets.reduce((sum, item) => sum + Number(item.usedHours || 0), 0);
  const totalBudget = data.roleBudgets.reduce((sum, item) => sum + Number(item.budgetHours || 0), 0);
  byId("budget-note").textContent = overBudget
    ? "At least one role exceeds its funded bank and needs review."
    : `All role banks pass. Total use is ${pct.format(totalUsed / Math.max(1, totalBudget))} of funded overflow capacity.`;
  renderKpis(data.kpis);
  renderComparison(data.comparison);
  renderBudgets(data.roleBudgets);
  renderActions(data.topActions);
  renderGaps(data.remainingGaps);
}

loadDemo().catch((error) => {
  byId("case-subtitle").textContent = "Demo data could not be loaded. Run demo/build_demo_data.py first, then serve this folder over HTTP.";
  console.error(error);
});
