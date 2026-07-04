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

function lerp(a, b, t) {
  return Number(a || 0) + (Number(b || 0) - Number(a || 0)) * t;
}

function interpolateRows(leftRows, rightRows, keyField, fields, t) {
  const rightByKey = new Map(rightRows.map((item) => [item[keyField], item]));
  return leftRows.map((left) => {
    const right = rightByKey.get(left[keyField]) || left;
    const out = { ...left };
    fields.forEach((field) => {
      out[field] = lerp(left[field], right[field], t);
    });
    return out;
  });
}

function pressureSample(data, multiplier) {
  const samples = [...(data.pressureScenarios || [])].sort((a, b) => a.pressureMultiplier - b.pressureMultiplier);
  if (!samples.length) {
    return {
      pressureMultiplier: data.parameters.surgeMultiplier / 100,
      kpis: {
        shortageAvoided: Number(data.kpis[0].value),
        coverageRiskAvoided: Number(data.kpis[1].value),
        manualShortage: data.parameters.baselineManualShortageHours,
        optimizedShortage: data.parameters.baseOptimizedShortageHours,
        budgetHours: data.parameters.baseBudgetHours,
        usedHours: data.parameters.baseUsedHours,
      },
      comparison: data.comparison,
      roleBudgets: data.roleBudgets,
      remainingGaps: data.remainingGaps,
      optimizedSchedule: data.optimizedSchedule,
    };
  }
  if (multiplier <= samples[0].pressureMultiplier) return samples[0];
  if (multiplier >= samples[samples.length - 1].pressureMultiplier) return samples[samples.length - 1];
  const rightIndex = samples.findIndex((sample) => sample.pressureMultiplier >= multiplier);
  const left = samples[rightIndex - 1];
  const right = samples[rightIndex];
  const t = (multiplier - left.pressureMultiplier) / Math.max(0.01, right.pressureMultiplier - left.pressureMultiplier);
  return {
    ...left,
    pressureMultiplier: multiplier,
    pressureRegime: t < 0.5 ? left.pressureRegime : right.pressureRegime,
    kpis: {
      shortageAvoided: lerp(left.kpis.shortageAvoided, right.kpis.shortageAvoided, t),
      coverageRiskAvoided: lerp(left.kpis.coverageRiskAvoided, right.kpis.coverageRiskAvoided, t),
      waitRiskReduced: lerp(left.kpis.waitRiskReduced, right.kpis.waitRiskReduced, t),
      manualShortage: lerp(left.kpis.manualShortage, right.kpis.manualShortage, t),
      optimizedShortage: lerp(left.kpis.optimizedShortage, right.kpis.optimizedShortage, t),
      budgetHours: lerp(left.kpis.budgetHours, right.kpis.budgetHours, t),
      usedHours: lerp(left.kpis.usedHours, right.kpis.usedHours, t),
      manualCoverageRisk: lerp(left.kpis.manualCoverageRisk, right.kpis.manualCoverageRisk, t),
      optimizedCoverageRisk: lerp(left.kpis.optimizedCoverageRisk, right.kpis.optimizedCoverageRisk, t),
    },
    comparison: interpolateRows(left.comparison, right.comparison, "mode", ["shortageHours", "coverageRiskHours", "waitRiskProxy"], t),
    roleBudgets: interpolateRows(left.roleBudgets, right.roleBudgets, "role", ["budgetHours", "usedHours", "remainingHours", "usedPct"], t),
    remainingGaps: interpolateRows(left.remainingGaps, right.remainingGaps, "role", ["remainingGapHours", "pressureHours"], t),
    optimizedSchedule: t < 0.5 ? left.optimizedSchedule : right.optimizedSchedule,
  };
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
  const sample = pressureSample(data, params.pressure / 100);
  const budgetFactor = params.budgetPct / base.overflowBudgetPct;
  const selectivityFactor = Math.max(0.68, Math.min(1.18, Math.sqrt(base.minimumCoveredGapHours / params.minGap)));
  const shiftFactor = params.shiftHours / base.overflowShiftHours;
  const shiftCoverageFactor = Math.max(0.72, Math.min(1.12, Math.sqrt(shiftFactor)));
  const coverageFactor = Math.max(0.35, Math.min(1.35, Math.pow(budgetFactor, 0.72) * selectivityFactor * shiftCoverageFactor));
  const manualShortage = sample.kpis.manualShortage;
  const shortageAvoided = Math.min(manualShortage, Math.max(0, sample.kpis.shortageAvoided * coverageFactor));
  const riskAvoided = Math.min(sample.kpis.manualCoverageRisk || 0, Math.max(0, sample.kpis.coverageRiskAvoided * coverageFactor));
  const budgetHours = sample.kpis.budgetHours * budgetFactor;
  const calloutFactor = Math.max(0.72, Math.min(1.22, Math.sqrt(base.minimumCoveredGapHours / params.minGap)));
  const shiftUseFactor = Math.max(0.86, Math.min(1.12, 0.92 + 0.08 * shiftFactor));
  const neededHours = sample.kpis.usedHours * calloutFactor * shiftUseFactor;
  const usedHours = Math.min(neededHours, budgetHours);
  const optimizedShortage = Math.max(0, manualShortage - shortageAvoided);
  const capacityUseFactor = usedHours / Math.max(1, sample.kpis.usedHours);
  const gapMultiplier = Math.max(
    manualShortage > 0 ? 0.08 : 0,
    Math.min(2.4, 1 / Math.max(0.55, coverageFactor * Math.sqrt(Math.max(0.55, capacityUseFactor)))),
  );
  const constrainedAndWeak = usedHours >= budgetHours * 0.98 && manualShortage > 0;
  const status = manualShortage <= 0 || (!constrainedAndWeak && shortageAvoided > sample.kpis.shortageAvoided * 0.55) ? "Ready" : "Review";
  return { ...sample, shortageAvoided, riskAvoided, usedHours, budgetHours, optimizedShortage, status, gapMultiplier };
}

function renderMetrics(data, scenario) {
  byId("metric-shortage").textContent = `${fmt.format(Math.round(scenario.shortageAvoided))}h`;
  byId("metric-shortage-note").textContent = `Original ${fmt.format(scenario.kpis.manualShortage)}h -> optimized ${fmt.format(scenario.optimizedShortage)}h`;
  byId("metric-risk").textContent = `${fmt.format(Math.round(scenario.riskAvoided))}h`;
  byId("metric-budget").textContent = pct.format(scenario.usedHours / Math.max(1, scenario.budgetHours));
  byId("metric-budget-note").textContent = `${fmt.format(scenario.usedHours)}h used of ${fmt.format(scenario.budgetHours)}h funded`;
  byId("metric-status").textContent = scenario.status;
  byId("metric-status-note").textContent = scenario.status === "Ready" ? "Ready for scheduler review" : "Adjust parameters before rollout";
}

function renderSchedule(id, rows, optimized = false) {
  if (!rows.length) {
    const colspan = optimized ? 6 : 6;
    byId(id).innerHTML = `<tr><td colspan="${colspan}" class="empty-row">${optimized ? "No overflow shifts recommended for this pressure level." : "No schedule rows available."}</td></tr>`;
    return;
  }
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
  const items = scenario.comparison.map((item) => ({ ...item }));
  const optimized = items.find((item) => item.mode === "Optimized overflow");
  if (optimized) {
    optimized.shortageHours = Math.round(scenario.optimizedShortage);
    optimized.coverageRiskHours = Math.max(0, Math.round(scenario.kpis.manualCoverageRisk - scenario.riskAvoided));
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
  const baseBudget = scenario.kpis.budgetHours || 1;
  const baseUsed = scenario.kpis.usedHours || 1;
  const budgetScale = scenario.budgetHours / baseBudget;
  const usedScale = scenario.usedHours / baseUsed;
  byId("budget-chart").innerHTML = scenario.roleBudgets
    .map((item) => {
      const budget = item.budgetHours * budgetScale;
      const used = Math.min(item.usedHours * usedScale, budget);
      const usedPct = (used / Math.max(1, budget)) * 100;
      return `<div class="meter">
        <div class="meter__top"><strong>${item.role}</strong><span>${fmt.format(used)}h used of ${fmt.format(budget)}h funded</span></div>
        <div class="meter__track"><b style="width:${Math.min(100, usedPct)}%"></b></div>
      </div>`;
    })
    .join("");
}

function renderGaps(data, scenario) {
  const adjusted = scenario.remainingGaps.map((item) => ({
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
  renderSchedule("optimized-schedule", scenario.optimizedSchedule, true);
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
  const anchors = (demoData.pressureScenarios || []).map((item) => `${Number(item.pressureMultiplier).toFixed(2)}x`).join(", ");
  byId("data-source").textContent = `Scenario: ${demoData.scenario}; pressure anchors: ${anchors}; source: ${demoData.generatedFrom}`;
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
