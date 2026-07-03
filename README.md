# Hospital Schedule Optimizer


<p align="center">
  <img alt="Python 3.11+" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white">
  <img alt="Tests included" src="https://img.shields.io/badge/tests-included-2ea44f">
  <img alt="No PHI synthetic data" src="https://img.shields.io/badge/data-synthetic%20%7C%20no%20PHI-0f766e">
  <img alt="Role aware optimization" src="https://img.shields.io/badge/optimizer-role--aware-7c3aed">
  <img alt="Budget guardrails" src="https://img.shields.io/badge/budget-guardrails-f59e0b">
  <img alt="License MIT" src="https://img.shields.io/badge/license-MIT-blue">
</p>

<p align="center"><strong>Safety-first ER workforce optimization with fixed rotations, role-specific demand, and funded overflow guardrails.</strong></p>

A synthetic evaluation harness and prototype for a **Hospital Schedule Optimizer**: a safety-first scheduling system that forecasts ER demand, preserves fixed workforce rotations, and recommends targeted overflow/on-call deployments.

Project path:
`/Users/antvibe/Documents/TabFM test/tabfm-healthcare-eval`

## 1. Solution overview

Hospital schedules are usually built from fixed rotations. Emergency departments typically have established physician, nurse practitioner, registered nurse, and support-worker lines. These rotations cannot be casually rearranged without creating fatigue, labour-relations, credentialing, continuity, or availability problems.

This solution respects that operating model. It does not try to rebuild the whole schedule from scratch. Instead, it asks a narrower and more practical question:

> Given the fixed schedule we already have, where should we deploy limited overflow capacity so the ER is safer and flows better?

The optimizer works in three steps:

1. Forecast role-specific demand for each facility-hour.
2. Compare forecast demand against the fixed core schedule.
3. Select overflow/on-call deployments that cover the most important remaining gaps while staying within funded role-specific overflow banks.

The current workflow models four workforce groups:

| Role | Meaning | Coverage rule |
| --- | --- | --- |
| MD | Emergency physician coverage | MD overflow and hospitalist overflow can cover MD-equivalent gaps |
| NP | Nurse practitioner coverage | NP gaps require NP-capable overflow |
| RN | Registered nurse coverage | RN gaps require RN-capable overflow |
| HCW | Healthcare worker/support coverage | HCW gaps require HCW-capable overflow |

The system is designed for advisory and shadow-mode use first. It produces a planned overflow schedule, audit metrics, and a remaining-capacity-needed table so operational leaders can see both the recommendation and the unresolved pressure.

## 2. What is novel about the solution

The novelty is not simply forecasting ER volume. Many tools can forecast demand. The important design choice is connecting demand forecasts to role-aware, budget-aware schedule recommendations.

Key differentiators:

| Capability | Why it matters |
| --- | --- |
| Fixed rotations are protected | The optimizer does not destabilize the standing schedule. It only recommends overflow deployments. |
| Role-specific demand | MD, NP, RN, and HCW needs are forecast and evaluated separately. A total provider count is not enough. |
| Separate overflow banks | Each role has its own funded overflow budget. Unused MD capacity cannot silently cover RN or HCW needs. |
| Multi-scenario forecasting | The same optimizer can evaluate TabFM-style, statistical ensemble, and conservative peak scenarios. |
| Manual baseline comparison | The solution compares optimized overflow against a traditional threshold-triggered manual process. |
| Remaining-gap transparency | The workflow reports where shortage remains after optimization rather than hiding it. |
| Operationally readable KPIs | Outputs use terms such as shortage hours and coverage-risk hours, not punitive language. |

This makes the project a scheduling optimizer, not just a predictive model demo.

## 3. Intended use

The solution is intended for health-system teams that already maintain provider rotations and overflow/on-call capacity.

Primary users:

- ER operations leaders;
- medical directors;
- nursing and allied-health operations leaders;
- scheduling teams;
- capacity and patient-flow teams;
- analytics teams validating forecast and optimization performance.

Recommended initial use:

1. Run in shadow mode for 60-90 days.
2. Compare optimized recommendations with the actual manual overflow process.
3. Measure shortage hours, coverage-risk hours, wait-risk proxy, manual activation misses, and provider acceptance.
4. Review remaining gaps by role and facility-hour.
5. Decide whether to move from advisory recommendations to operational scheduling support.

The solution should not be positioned as a base labour reduction tool. Its strongest first use is improving safety, flow, and the value of already-funded overflow capacity.

## 4. Architecture

The workflow has two separate layers:

1. Forecasting layer: predicts future role-specific demand from ER operations data.
2. Optimization layer: converts forecasts into concrete overflow deployment recommendations.

TabFM can be used in the forecasting layer, but it is not the optimizer. The optimizer is a separate scheduling decision layer.

```text
raw ER operations + fixed rotations + resource state
        -> forecast scenarios
        -> role-specific funded overflow optimizer
        -> planned schedule table
        -> audit, comparison, and report
```

The optimizer is intentionally constrained:

- core rotations are locked;
- only overflow/on-call candidates can be selected;
- 10-hour overflow shifts are used by default;
- MD, NP, RN, and HCW needs are tracked separately;
- hospitalist overflow counts only as `0.75` MD-equivalent capacity;
- unused capacity in one role cannot be spent on another role;
- remaining shortage is reported by role and by hour.

## 5. Dataset and semantic tables

The solution starts from realistic ER operations tables, not from a pre-summarized staffing answer. The synthetic data is mixed-grain, like real hospital operations data.

| Mnemonic | Table | Grain | Meaning |
| --- | --- | --- | --- |
| EVH | `er_visits_historical.parquet` | Visit-level | ER visit demand history |
| PSH | `provider_shifts_historical.parquet` | Provider-shift-level | Fixed rotations and overflow candidates |
| RAV | `resource_availability.parquet` | Facility-hour-level | Beds, rooms, wait-risk, on-call pool, environmental flags |
| USS | `upcoming_schedule_skeleton.parquet` | Facility-hour-level | Future schedule shell |
| PTD | `predictions.tomorrow_demands.parquet` | Facility-hour-scenario-level | Forecast demand scenarios |
| SRR | `metadata.system_rollout_registry.parquet` | Metadata | Rollout control metadata |
| PSP | `planned_schedule_table.parquet` | Activation-event-level | Optimizer-selected overflow deployments |
| ASS | `actual_system_state.parquet` | Facility-hour-level | Realized synthetic operating state for audit |

### Visit-level demand: EVH

Each ER visit is represented as a row. In a production setting this would come from ED/EHR operational systems.

Core fields:

- `timestamp`
- `facility_id`
- `department_id`
- `visit_id`
- `acuity_score`
- `disposition`
- `environmental_flags`

The visit table is deliberately raw enough for feature engineering. Hourly demand and pressure are computed downstream rather than pre-baked.

### Provider scheduling history: PSH

`PSH` represents both protected core rotations and selectable overflow candidates.

Important fields:

- `shift_id`
- `provider_id`
- `facility_id`
- `department_id`
- `shift_start`, `shift_end`
- `rotation_line_id`
- `role`
- `is_core_rotation`
- `is_overflow_candidate`
- `activation_capacity_md_equiv`
- `activation_capacity_np_equiv`
- `activation_capacity_rn_equiv`
- `activation_capacity_hcw_equiv`

Core rotations are the standing schedule. Overflow candidates are available capacity that may or may not be activated.

### Actual and predicted role demand: ASS and PTD

The audit table includes realized role-specific need:

- `actual_required_md`
- `actual_required_np`
- `actual_required_rn`
- `actual_required_hcw`

Forecast scenarios include matching role-specific predictions:

- `predicted_required_md`
- `predicted_required_np`
- `predicted_required_rn`
- `predicted_required_hcw`

Role-level demand is essential. A total provider forecast is not enough because MD, NP, RN, and HCW capacity are not interchangeable.

## 6. Workforce rotations and overflow lines

### Protected core rotations

| Role | Rotation line | Time | Purpose |
| --- | --- | --- | --- |
| MD | `MD_TRAUMA_DAY` | 07:00-17:00 | Trauma/acute lead |
| MD | `MD_FAST_TRACK` | 09:00-19:00 | Fast-track physician |
| NP | `NP_SWING` | 13:00-23:00 | Swing NP |
| MD | `MD_NIGHT` | 21:00-07:00 | Overnight physician |
| RN | `RN_DAY_A` | 07:00-19:00 | Main RN day coverage |
| RN | `RN_DAY_B` | 09:00-21:00 | Peak/day overlap RN |
| RN | `RN_NIGHT` | 19:00-07:00 | Overnight RN coverage |
| HCW | `HCW_DAY` | 07:00-17:00 | Daytime support |
| HCW | `HCW_EVE` | 13:00-23:00 | Evening support |
| HCW | `HCW_NIGHT` | 21:00-07:00 | Overnight support |

### Selectable overflow candidates

| Role | Overflow line | Time | Coverage |
| --- | --- | --- | --- |
| MD | `ON_CALL_MD` | 10-hour block | 1.0 MD |
| MD | `ON_CALL_MD_EVE` | 10-hour block | 1.0 MD |
| NP | `ON_CALL_NP_SWING` | 10-hour block | 1.0 NP |
| NP | `ON_CALL_NP_EVE` | 10-hour block | 1.0 NP |
| MD | `HOSPITALIST_OVERFLOW` | 10-hour block | 0.75 MD |
| RN | `ON_CALL_RN_DAY` | 10-hour block | 1.0 RN |
| RN | `ON_CALL_RN_EVE` | 10-hour block | 1.0 RN |
| RN | `ON_CALL_RN_NIGHT` | 10-hour block | 1.0 RN |
| HCW | `ON_CALL_HCW_DAY` | 10-hour block | 1.0 HCW |
| HCW | `ON_CALL_HCW_EVE` | 10-hour block | 1.0 HCW |

## 7. Separate funded overflow banks

Each role has its own overflow budget:

```text
MD overflow bank = core MD hours * 10%
NP overflow bank = core NP hours * 10%
RN overflow bank = core RN hours * 10%
HCW overflow bank = core HCW hours * 10%
```

CLI flags:

- `--overflow-budget-share-md`
- `--overflow-budget-share-np`
- `--overflow-budget-share-rn`
- `--overflow-budget-share-hcw`

A run passes budget control only if no role-specific bank is exceeded.

## 8. Forecast scenarios

The workflow evaluates three scenarios:

| Scenario | Purpose |
| --- | --- |
| `tabfm_guarded` | TabFM-style forecast path with guardrails and role-level demand fields |
| `statistical_ensemble` | Traditional statistical/ML comparison forecast |
| `conservative_peak` | High-side surge scenario for stress testing |

The scenario framework is intended to compare decision quality under different plausible futures, not to claim a single forecast is always correct.

## 9. Manual baseline vs optimized overflow

The manual baseline simulates a traditional rule-based overflow process:

- waiting room threshold;
- wait-risk threshold;
- ambulance surge;
- heatwave or flu-surge condition;
- manual call-out of an overflow provider.

Manual activations may miss because they are early, late, unavailable, or assigned to the wrong role/site. The optimized workflow uses forecasted role gaps to pre-place overflow where it is expected to cover the most need.

## 10. Run the workflow

```bash
cd "/Users/antvibe/Documents/TabFM test/tabfm-healthcare-eval"
python3 run_fixed_rotation_overflow_workflow.py \
  --seed 42 \
  --facilities 4 \
  --days 30 \
  --overflow-shift-hours 10 \
  --overflow-budget-share-md 0.10 \
  --overflow-budget-share-np 0.10 \
  --overflow-budget-share-rn 0.10 \
  --overflow-budget-share-hcw 0.10 \
  --demand-pressure-multiplier 1.35 \
  --min-overflow-covered-gap-hours 4.0 \
  --out outputs/fixed_rotation_overflow_multirole_funded10_v1 \
  --print-top-n 5
```

The run prints Top-N rows for every major table and writes reports.

## 11. Main outputs

Latest run root:
`/Users/antvibe/Documents/TabFM test/tabfm-healthcare-eval/outputs/fixed_rotation_overflow_multirole_funded10_v1`

Reports:

- `fixed_rotation_overflow_report.md`
- `fixed_rotation_overflow_report.html`

Key output tables:

| File | Purpose |
| --- | --- |
| `overflow_reallocation_summary.csv` | Scenario-level KPI summary with role budgets, usage, and reserves |
| `manual_vs_optimized_overflow.csv` | Static vs manual vs optimized comparison |
| `recommended_overflow_capacity_needed_by_hour.csv` | Remaining uncovered MD/NP/RN/HCW demand by hour |
| `planned_schedule_table.parquet` | Optimizer-selected overflow deployments |
| `manual_overflow_activation_log.parquet` | Simulated manual overflow activations |

## 12. Illustration: multi-role funded-10% case study

This case study illustrates the solution on a synthetic 4-facility, 30-day ER network. It is not a claim about a specific real hospital. It shows how the optimizer behaves when each workforce group has a separate 10% funded overflow bank.

### Overall result

| Metric | Manual overflow | Optimized overflow | Improvement |
| --- | ---: | ---: | ---: |
| Total shortage hours | 877 | 381 | 496 fewer hours |
| Coverage-risk hours | 704 | 354 | 350 fewer hours |
| Wait-risk proxy during shortage | 396.65 | 204.95 | 191.70 lower |
| MD shortage hours | 199 | 89 | 110 fewer hours |
| NP shortage hours | 121 | 78 | 43 fewer hours |
| RN shortage hours | 311 | 114 | 197 fewer hours |
| HCW shortage hours | 246 | 100 | 146 fewer hours |

### Role-specific funded overflow usage

| Role | Core hours | Funded 10% bank | Optimized used | Reserve remaining |
| --- | ---: | ---: | ---: | ---: |
| MD | 3,640.0 | 364.0 | 290.0 | 74.0 |
| NP | 1,200.0 | 120.0 | 120.0 | 0.0 |
| RN | 4,368.0 | 436.8 | 430.0 | 6.8 |
| HCW | 3,640.0 | 364.0 | 360.0 | 4.0 |
| Total | 12,848.0 | 1,284.8 | 1,200.0 | 84.8 |

Interpretation: the optimizer uses most of the funded role-specific overflow capacity without exceeding any role bank. RN and HCW pressure are material in this scenario, which is operationally realistic because nursing and support capacity often drive flow constraints.

## 13. ROI framing for BC-style use

The case assumes each role already has a funded 10% overflow bank. Therefore, the first business question is not, "Can we add new hours?" It is:

> Can we recover more value from overflow capacity the system has already funded?

The illustration shows meaningful operational improvement versus manual overflow:

- `496` fewer shortage hours;
- `350` fewer coverage-risk hours;
- all role-specific overflow banks stayed within 10%;
- total reserve remaining was `84.8` role-hours.

A conservative business case should replace the synthetic values with local Finance and Medical Affairs assumptions. The strongest framing is safety, flow, and funded-capacity optimization rather than base labour reduction.

## 14. Assumptions and governance notes

- All data is synthetic.
- No PHI is used.
- HCW means healthcare worker/support capacity such as care aide, patient transport/support, or ED support worker in this synthetic model.
- RN and HCW cannot substitute for MD or NP coverage.
- MD/NP cannot substitute for RN/HCW workflow load.
- Hospitalist overflow remains `0.75` MD-equivalent only.
- Forecasts should be monitored for drift and flat prediction behavior.
- Recommendations should be reviewed against local collective agreements, credentialing rules, fatigue rules, site policies, and provider availability.
- Production use should start in advisory shadow mode before active scheduling.
