const fmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });
const pct = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1, style: "percent" });

let demoData = null;
let runCount = 0;

function byId(id) {
  return document.getElementById(id);
}

function maxOf(items, fields) {
  return Math.max(1, ...items.flatMap((item) => fields.map((field) => Number(item[field]) || 0)));
}

function numberValue(id) {
  return Number(byId(id).value);
}

function getParams() {
  return {
    budgetPct: numberValue("budget-control"),
    pressure: numberValue("pressure-control"),
    minGap: numberValue("gap-control"),
    shiftHours: numberValue("shift-control"),
  };
}

function updateControlLabels() {
  const p = getParams();
  byId("budget-value").textContent = `${p.budgetPct}%`;
  byId("pressure-value").textContent = `${(p.pressure / 100).toFixed(2)}x`;
  byId("gap-value").textContent = `${p.minGap}h`;
  byId("shift-value").textContent = `${p.shiftHours}h`;
}

function simulateScenario(data, params) {
  const base = data.parameters;
  const budgetFactor = params.budgetPct / base.overflowBudgetPct;
  const pressureFactor = params.pressure / base.surgeMultiplier;
  const selectivityFactor = Math.max(0.68, Math.min(1.18, Math.sqrt(base.minimumCoveredGapHours / params.minGap)));
  const shiftFactor = params.shiftHours / base.overflowShiftHours;
  const shiftCoverageFactor = Math.max(0.72, Math.min(1.12, Math.sqrt(shiftFactor)));
  const coverageFactor = Math.max(0.35, Math.min(1.35, Math.pow(budgetFactor, 0.72) * selectivityFactor * shiftCoverageFactor));
  const shortageAvoidedBase = Number(data.kpis[0].value);
  const riskAvoidedBase = Number(data.kpis[1].value);
  const manualShortage = base.baselineManualShortageHours * pressureFactor;
  const shortageAvoided = Math.min(manualShortage * 0.86, Math.max(0, shortageAvoidedBase * coverageFactor * Math.sqrt(pressureFactor)));
  const riskAvoided = Math.min(data.comparison[1].coverageRiskHours * pressureFactor * 0.9, Math.max(0, riskAvoidedBase * coverageFactor * Math.sqrt(pressureFactor)));
  const budgetHours = base.baseBudgetHours * budgetFactor;
  const calloutFactor = Math.max(0.72, Math.min(1.22, Math.sqrt(base.minimumCoveredGapHours / params.minGap)));
  const neededHours = base.baseUsedHours * pressureFactor * calloutFactor * shiftFactor;
  const usedHours = Math.min(neededHours, budgetHours);
  const optimizedShortage = Math.max(0, manualShortage - shortageAvoided);
  const gapMultiplier = Math.max(0.18, Math.min(2.4, Math.pow(pressureFactor, 1.25) / Math.max(0.45, coverageFactor)));
  const constrainedAndWeak = usedHours >= budgetHours * 0.98 && shortageAvoided < shortageAvoidedBase * 0.85;
  const status = !constrainedAndWeak && shortageAvoided > shortageAvoidedBase * 0.55 ? "Ready" : "Review";
  return { shortageAvoided, riskAvoided, usedHours, budgetHours, optimizedShortage, status, gapMultiplier };
}

function renderMetrics(data, scenario) {
  byId("metric-shortage").textContent = `${fmt.format(Math.round(scenario.shortageAvoided))}h`;
  byId("metric-shortage-note").textContent = `Original ${fmt.format(data.parameters.baselineManualShortageHours)}h -> optimized ${fmt.format(scenario.optimizedShortage)}h`;
  byId("metric-risk").textContent = `${fmt.format(Math.round(scenario.riskAvoided))}h`;
  byId("metric-budget").textContent = pct.format(scenario.usedHours / Math.max(1, scenario.budgetHours));
  byId("metric-budget-note").textContent = `${fmt.format(scenario.usedHours)} of ${fmt.format(scenario.budgetHours)} funded hours`;
  byId("metric-status").textContent = scenario.status;
  byId("metric-status-note").textContent = scenario.status === "Ready" ? "Ready for scheduler review" : "Adjust parameters before rollout";
}

function renderSchedule(id, rows, optimized = false) {
  byId(id).innerHTML = rows
    .map((row) => {
      if (optimized) {
        return `<tr>
          <td data-label="Facility">${row.facility}</td>
          <td data-label="Role">${row.role}</td>
          <td data-label="Start">${row.start}</td>
          <td data-label="End">${row.end}</td>
          <td data-label="Priority"><span class="pill">${row.priority}</span></td>
          <td data-label="Score">${fmt.format(row.score)}</td>
        </tr>`;
      }
      return `<tr>
        <td data-label="Facility">${row.facility}</td>
        <td data-label="Role">${row.role}</td>
        <td data-label="Line">${row.line}</td>
        <td data-label="Start">${row.start}</td>
        <td data-label="End">${row.end}</td>
        <td data-label="Hours">${fmt.format(row.hours)}</td>
      </tr>`;
    })
    .join("");
}

function renderComparison(data, scenario) {
  const items = data.comparison.map((item) => ({ ...item }));
  const optimized = items.find((item) => item.mode === "Optimized overflow");
  if (optimized) {
    optimized.shortageHours = Math.round(scenario.optimizedShortage);
    optimized.coverageRiskHours = Math.max(0, Math.round(data.comparison[1].coverageRiskHours - scenario.riskAvoided));
  }
  const max = maxOf(items, ["shortageHours", "coverageRiskHours"]);
  byId("comparison-chart").innerHTML = items
    .map((item) => {
      const shortagePct = (Number(item.shortageHours) / max) * 100;
      const riskPct = (Number(item.coverageRiskHours) / max) * 100;
      return `<div class="bar-row">
        <div class="bar-row__label">${item.mode}</div>
        <div class="bar-stack">
          <div class="bar-line"><span>Shortage</span><div class="bar-track"><b style="width:${shortagePct}%"></b></div><strong>${fmt.format(item.shortageHours)}</strong></div>
          <div class="bar-line bar-line--risk"><span>Risk</span><div class="bar-track"><b style="width:${riskPct}%"></b></div><strong>${fmt.format(item.coverageRiskHours)}</strong></div>
        </div>
      </div>`;
    })
    .join("");
}

function renderBudgets(data, scenario) {
  const budgetScale = scenario.budgetHours / data.parameters.baseBudgetHours;
  const usedScale = scenario.usedHours / data.parameters.baseUsedHours;
  byId("budget-chart").innerHTML = data.roleBudgets
    .map((item) => {
      const budget = item.budgetHours * budgetScale;
      const used = Math.min(item.usedHours * usedScale, budget);
      const usedPct = (used / Math.max(1, budget)) * 100;
      return `<div class="meter">
        <div class="meter__top"><strong>${item.role}</strong><span>${fmt.format(used)} / ${fmt.format(budget)}h</span></div>
        <div class="meter__track"><b style="width:${Math.min(100, usedPct)}%"></b></div>
      </div>`;
    })
    .join("");
}

function renderGaps(data, scenario) {
  const adjusted = data.remainingGaps.map((item) => ({
    ...item,
    remainingGapHours: Math.max(0, item.remainingGapHours * scenario.gapMultiplier),
  }));
  const max = maxOf(adjusted, ["remainingGapHours"]);
  byId("gaps-chart").innerHTML = adjusted
    .map((item) => {
      const width = (Number(item.remainingGapHours) / max) * 100;
      return `<article class="gap-card">
        <strong>${item.role}</strong>
        <div class="bar-track"><b style="width:${width}%"></b></div>
        <span>${fmt.format(item.remainingGapHours)} uncovered role-hours</span>
        <small>${item.pressureHours} hours with a remaining gap</small>
      </article>`;
    })
    .join("");
}

function renderAll(regenerated = false) {
  if (!demoData) return;
  updateControlLabels();
  const scenario = simulateScenario(demoData, getParams());
  renderMetrics(demoData, scenario);
  renderSchedule("initial-schedule", demoData.initialSchedule);
  renderSchedule("optimized-schedule", demoData.optimizedSchedule, true);
  renderComparison(demoData, scenario);
  renderBudgets(demoData, scenario);
  renderGaps(demoData, scenario);
  if (regenerated) {
    runCount += 1;
    byId("run-stamp").textContent = `Run ${runCount} regenerated`;
  }
}

async function loadDemo() {
  const response = await fetch("data/demo_case.json");
  if (!response.ok) throw new Error(`Unable to load demo data: ${response.status}`);
  demoData = await response.json();
  byId("case-subtitle").textContent = demoData.case.subtitle;
  byId("story-problem").textContent = demoData.plainLanguage.problem;
  byId("story-manual").textContent = demoData.plainLanguage.manual;
  byId("story-optimizer").textContent = demoData.plainLanguage.optimizer;
  byId("story-remaining").textContent = demoData.plainLanguage.remaining;
  byId("data-source").textContent = `Scenario: ${demoData.scenario}; source: ${demoData.generatedFrom}`;
  ["budget-control", "pressure-control", "gap-control", "shift-control"].forEach((id) => {
    byId(id).addEventListener("input", updateControlLabels);
  });
  byId("regen-button").addEventListener("click", () => renderAll(true));
  renderAll(false);
}

loadDemo().catch((error) => {
  byId("case-subtitle").textContent = "Demo data could not be loaded. Run demo/build_demo_data.py, then serve the demo over HTTP.";
  console.error(error);
});
