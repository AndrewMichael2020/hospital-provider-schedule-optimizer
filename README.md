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

<p align="center" style="color: #d1242f;"><strong>Live demo: <a href="https://andrewmichael2020.github.io/hospital-provider-schedule-optimizer/">Hospital Schedule Optimizer scheduling console</a></strong></p>

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

## 15. Implementation pathway: dashboard, CSV, and future integration

The recommended first deployment is **dashboard plus CSV**, not direct schedule write-back. This gives provider schedulers and ER operations leaders practical recommendations while keeping the workflow safe, auditable, and easy to reverse.

### Recommended first operating model

Use two outputs in parallel:

| Output | Primary audience | Purpose |
| --- | --- | --- |
| Dashboard | ER operations leaders, staffing office leads, medical/nursing leadership | Review pressure, recommendations, role-specific gaps, budget use, and outcomes |
| CSV/worklist | Provider schedulers, staffing clerks, charge/support roles | Act on recommendations in existing scheduling workflows |

The dashboard explains the recommendation. The CSV makes it actionable.

### Why not direct write-back first

Direct write-back into scheduling systems should come later. Early write-back creates avoidable risk:

- local scheduling systems may be the source of truth for different roles;
- physician, NP, RN, and HCW schedules may live in different tools;
- fatigue, union, credentialing, and local availability rules may not be fully represented at first;
- a recommendation may be clinically reasonable but operationally impossible;
- schedulers need a period of trust-building before automation changes live schedules.

For the first production-style pilot, the optimizer should recommend; humans should decide.

## 16. CSV implementation package

A practical deployment should generate a small set of stable CSV files every day. These can be delivered by SFTP, secure shared folder, SharePoint, dashboard download, or data warehouse table export.

### 16.1 Scheduler action CSV

File:
`recommended_overflow_actions.csv`

Purpose: the scheduler's main action list.

Recommended columns:

| Column | Meaning |
| --- | --- |
| `recommendation_id` | Stable unique recommendation ID |
| `run_id` | Optimizer run that created the recommendation |
| `scenario_name` | Forecast scenario, such as `tabfm_guarded` or `conservative_peak` |
| `facility_id` | Site/facility |
| `department_id` | Department or unit, such as `ER_MAIN` |
| `role` | `MD`, `NP`, `RN`, or `HCW` |
| `recommended_start` | Recommended shift start |
| `recommended_end` | Recommended shift end |
| `duration_hours` | Recommended duration, usually 10 hours |
| `priority_tier` | `P1`, `P2`, or `P3` |
| `priority_score` | Numeric ranking score |
| `reason` | Plain-language explanation of the recommendation |
| `expected_shortage_reduction_hours` | Expected role-hours covered |
| `expected_coverage_risk_reduction` | Expected reduction in coverage-risk hours |
| `role_budget_hours` | Funded overflow budget for the role |
| `role_budget_used_after` | Budget used if recommendation is accepted |
| `role_budget_remaining_after` | Remaining role budget after recommendation |
| `candidate_provider_id` | Optional if provider-specific recommendation is available |
| `scheduler_status` | `pending`, `accepted`, `rejected`, or `modified` |
| `scheduler_comment` | Optional human note |

Minimum rule: every row must have a stable `recommendation_id`, role, facility, start, end, and priority.

### 16.2 Remaining capacity-needed CSV

File:
`recommended_overflow_capacity_needed_by_hour.csv`

Purpose: show unresolved demand after optimization.

Recommended columns:

| Column | Meaning |
| --- | --- |
| `facility_id` | Site/facility |
| `timestamp_hour` | Hour of remaining pressure |
| `department_id` | Unit/department |
| `actual_required_md` | Realized or simulated MD need |
| `actual_required_np` | Realized or simulated NP need |
| `actual_required_rn` | Realized or simulated RN need |
| `actual_required_hcw` | Realized or simulated HCW need |
| `core_md_coverage` | Fixed MD coverage |
| `core_np_coverage` | Fixed NP coverage |
| `core_rn_coverage` | Fixed RN coverage |
| `core_hcw_coverage` | Fixed HCW coverage |
| `optimized_overflow_md_coverage` | Optimized MD overflow coverage |
| `optimized_overflow_np_coverage` | Optimized NP overflow coverage |
| `optimized_overflow_rn_coverage` | Optimized RN overflow coverage |
| `optimized_overflow_hcw_coverage` | Optimized HCW overflow coverage |
| `remaining_md_gap` | Remaining MD gap |
| `remaining_np_gap` | Remaining NP gap |
| `remaining_rn_gap` | Remaining RN gap |
| `remaining_hcw_gap` | Remaining HCW gap |
| `priority_tier` | Remaining-gap priority |
| `scenario_name` | Forecast scenario |

This table is important for leadership. It shows what the funded overflow bank could not solve.

### 16.3 Scheduler decision audit CSV

File:
`scheduler_decision_audit.csv`

Purpose: track whether recommendations were reviewed, accepted, modified, or rejected.

Recommended columns:

| Column | Meaning |
| --- | --- |
| `recommendation_id` | Links to `recommended_overflow_actions.csv` |
| `run_id` | Optimizer run ID |
| `reviewed_at` | Timestamp of scheduler review |
| `reviewed_by_role` | Staffing clerk, charge nurse, medical lead, etc. |
| `scheduler_status` | `pending`, `accepted`, `rejected`, or `modified` |
| `final_role` | Role after human review |
| `final_facility_id` | Facility after human review |
| `final_start` | Final approved start time |
| `final_end` | Final approved end time |
| `rejection_reason` | Reason if rejected |
| `modification_reason` | Reason if modified |
| `scheduler_comment` | Human-readable comment |

Recommended rejection reasons:

- `provider_unavailable`
- `already_covered`
- `clinical_need_changed`
- `too_late_to_call`
- `contract_or_fatigue_rule`
- `budget_or_bank_policy`
- `wrong_role_or_site`
- `scheduler_disagreed`
- `other`

These reasons are not administrative noise. They are model-improvement data.

### 16.4 Actual deployment reconciliation CSV

File:
`actual_deployment_reconciliation.csv`

Purpose: determine whether an accepted recommendation actually happened.

Recommended columns:

| Column | Meaning |
| --- | --- |
| `recommendation_id` | Links back to recommendation |
| `scheduled_shift_id` | Shift created or selected by scheduler |
| `actual_worked_shift_id` | Payroll/timekeeping worked-shift ID |
| `scheduled_provider_id` | Provider scheduled |
| `actual_provider_id` | Provider who actually worked |
| `scheduled_start` | Final scheduled start |
| `scheduled_end` | Final scheduled end |
| `actual_clock_in` | Timekeeping start |
| `actual_clock_out` | Timekeeping end |
| `deployment_status` | `worked`, `partial`, `cancelled`, `no_show`, or `replaced` |
| `actual_hours_worked` | Actual worked hours |
| `variance_minutes` | Timing difference from recommendation |

This table closes the loop between recommendation, scheduling decision, and actual worked hours.

## 17. Dashboard implementation

The first dashboard should be operational, not decorative. It should help leaders and schedulers answer: what should we do today, why, and did it work?

Recommended dashboard tabs:

| Tab | Purpose |
| --- | --- |
| Executive summary | Show shortage reduction, coverage-risk reduction, budget use, and rollout readiness |
| Scheduler worklist | Filterable recommendations by facility, role, time, and priority |
| Role view | MD, NP, RN, and HCW demand, gaps, and overflow use |
| Facility view | Site-level pressure and recommendations |
| Timeline view | Hour-by-hour forecast pressure and planned overflow |
| Remaining gaps | Uncovered capacity after optimization |
| Adoption funnel | Generated -> reviewed -> accepted/modified -> worked |
| Rejection reasons | Why recommendations were not implemented |
| Outcome audit | Compare implemented vs not implemented recommendations |

### Dashboard metrics

Recommended top-level KPIs:

| Metric | Meaning |
| --- | --- |
| `recommendations_generated` | Count of optimizer recommendations |
| `recommendations_reviewed` | Count reviewed by schedulers |
| `recommendations_accepted` | Count accepted as-is |
| `recommendations_modified` | Count modified by humans |
| `recommendations_rejected` | Count rejected |
| `recommendations_worked` | Count that actually resulted in worked shifts |
| `shortage_hours_reduced` | Estimated or realized shortage reduction |
| `coverage_risk_hours_reduced` | Estimated or realized reduction in coverage-risk hours |
| `overflow_budget_used_by_role` | Percent of each funded role bank used |
| `remaining_gap_hours_by_role` | Uncovered demand after optimization |

### Adoption funnel

The adoption funnel is the most important dashboard feature after the scheduler worklist.

```text
recommendation generated
        -> reviewed by scheduler
        -> accepted / modified / rejected
        -> scheduled
        -> actually worked
        -> outcome measured
```

Example metrics:

| Metric | Formula |
| --- | --- |
| Review rate | reviewed / generated |
| Acceptance rate | accepted / reviewed |
| Modification rate | modified / reviewed |
| Rejection rate | rejected / reviewed |
| Implementation rate | worked / accepted_or_modified |
| End-to-end adoption | worked / generated |
| Timing accuracy | worked within recommended window / worked |

Without this funnel, the team only knows what the optimizer recommended. With it, the team knows whether the recommendation was operationally adopted.

## 18. Maturity model

Use a staged maturity model. Do not jump directly to automated schedule writes.

| Level | Name | Output | Human role | Integration risk |
| --- | --- | --- | --- | --- |
| 0 | Offline analysis | Reports only | Analytics review | Very low |
| 1 | CSV shadow mode | Daily CSV recommendations | Scheduler compares manually | Low |
| 2 | Dashboard + CSV | Dashboard and action CSV | Scheduler reviews and records decisions | Low-moderate |
| 3 | Scheduler worklist | Review queue with accept/reject/modify | Scheduler actively manages recommendations | Moderate |
| 4 | Proposed schedule write-back | Pre-filled proposed shifts requiring approval | Scheduler approves before posting | Higher |
| 5 | Controlled direct write-back | Idempotent API updates to scheduling system | Human exception management | Highest |

Recommended pilot target: **Level 2**.

Recommended production target after trust is established: **Level 3 or Level 4**.

Only pursue Level 5 after governance, audit, rollback, identity matching, and idempotency controls are proven.

## 19. Look forward: idempotent integration

If the system eventually writes proposed shifts or tasks into another scheduling system, integration must be idempotent. This means the same optimizer run can be safely replayed without creating duplicate shifts, duplicate tasks, or conflicting records.

### Required identifiers

Every recommendation should include stable keys:

| Field | Purpose |
| --- | --- |
| `run_id` | Unique optimizer run |
| `recommendation_id` | Unique recommendation across runs |
| `source_system` | `hospital_schedule_optimizer` |
| `source_version` | Code/model/config version |
| `scenario_name` | Forecast scenario used |
| `facility_id` | Site |
| `role` | Workforce role |
| `recommended_start` | Recommendation start |
| `recommended_end` | Recommendation end |
| `idempotency_key` | Stable key for external write-back |

Recommended idempotency key pattern:

```text
hospital_schedule_optimizer::{run_date}::{scenario_name}::{facility_id}::{role}::{recommended_start_iso}::{recommended_end_iso}
```

For provider-specific recommendations, include `candidate_provider_id` in the key.

### Idempotent write behavior

When integrating with a scheduling system or worklist:

1. Check whether `idempotency_key` already exists.
2. If it exists and content is unchanged, do nothing.
3. If it exists and content changed, create a new version or update only allowed fields.
4. If it does not exist, create a new proposed task/shift.
5. Never create two active records with the same idempotency key.
6. Store external IDs returned by the target system.
7. Preserve a full audit trail of create/update/cancel actions.

### Recommended write-back statuses

Use advisory statuses before live schedule mutation:

| Status | Meaning |
| --- | --- |
| `proposed` | Optimizer created recommendation |
| `under_review` | Scheduler opened or claimed it |
| `accepted` | Scheduler accepted recommendation |
| `modified` | Scheduler changed it |
| `rejected` | Scheduler rejected it |
| `posted_to_schedule` | Recommendation became a scheduled shift |
| `worked` | Shift was confirmed by actual worked hours |
| `cancelled` | Shift was cancelled |

### Rollback and safety

Any future write-back integration should support:

- dry-run mode;
- shadow mode;
- single-facility pilot mode;
- role-by-role enablement;
- maximum number of recommendations per day;
- budget cap enforcement before write;
- manual approval before posting;
- immutable audit logs;
- safe cancellation or supersession of prior recommendations.

The safest near-term architecture is to write to an intermediate worklist, not directly to the scheduling source of truth.

## 20. Practical integration recommendation

For a Health Authority-wide or provincial Canadian deployment, the recommended sequence is:

1. Start with historical data extracts from EHR/ED operations, scheduling, and timekeeping.
2. Run daily shadow-mode optimizer jobs.
3. Produce dashboard plus CSV outputs.
4. Ask schedulers to record accept/reject/modify decisions.
5. Reconcile accepted recommendations against actual worked hours.
6. Review weekly adoption and outcome metrics.
7. Move to a scheduler worklist only after the CSV process is trusted.
8. Consider idempotent write-back only after governance approval and system-specific integration design.

This keeps the first implementation practical while preserving a clear path toward deeper integration later.
