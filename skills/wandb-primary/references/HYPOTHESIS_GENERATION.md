<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: skills
-->
# Synergistic Hypothesis Generation

When asked to analyze an experiment, diagnose unexpected results, or recommend a next
experiment, execute this four-phase scientific discovery loop. Each phase gates the next.
Do not skip phases or merge them into a single script.

---

## Phase 1: State the Prior

Before touching the data, explicitly state what a domain expert would expect given the
experimental setup described or inferred from the project name and run metadata.

Answer these questions before running any code:
- What metrics should **vary** across runs or groups? Which should be correlated?
- What does the method class (SSL, RL, SFT, CPT, sweep, fine-tune) predict about
  baseline behavior?
- What does "working as intended" look like and what would "broken" look like?

Write the prior in plain language. This is your null hypothesis. State it explicitly
in your response before the first script runs.

---

## Phase 2: Calculate the Surprise

Fetch a broad project overview — run counts, states, summary metrics, artifact names,
and Weave traces (if the project has them) — and compare the data against your Phase 1 prior.

Identify the **point of maximum Bayesian surprise**: the specific place where the data
most strongly violates your expectation. The surprise is not a trend that confirms your
prior; it is the deviation that contradicts it.

Two high-surprise patterns that always require a follow-up script:

**No variance where you expected differentiation.** If a metric is nearly identical
across all runs or groups you expected it to separate, either (a) the metric measures
something invariant rather than the per-run outcome you assumed, or (b) the segmentation
variable you expected is absent from the data. Do not accept a uniform metric as the
answer; chase the signal that does differentiate.

**Cross-dimensional contradiction within a specific element.** When one particular group,
source, or configuration is extreme on two correlated quality dimensions in opposite
directions (e.g., high on volume but low on quality, high acceptance but low verified
agreement), that specific element is almost certainly the mechanistic anomaly.

**Phase 2 is a free exploration; use it fully.** Your goal is to exhaust the project's
evidence, not to find the first explanation that fits. Run as many scripts as needed.
Every artifact, trace source, and metric slice is a potential lead. When you find a
valid finding at one level, treat it as a starting point, not a conclusion. The most
actionable anomaly is almost always one level more specific than your first observation:
not "the data has noise" but "source X specifically has noise." Keep digging until the
data stops producing new signals.

**Mechanism identification is not localization.** Identifying a class of problem is a
direction, not a finding. The next question is always: which specific element is
responsible? Which source, model, task type, or configuration? Until you can name that
element with evidence, you are still in Phase 2.

**"Checking" means fetching data, not listing names.** An artifact appears in the run
summary as a name. That name tells you the table exists. It does not tell you what the
data shows. You must download and read the actual rows. For each artifact table that shows
a per-group breakdown (by source, by class, by model, by configuration), extract at least
two quality metrics per group simultaneously. Cross-dimensional contradictions (e.g., high
acceptance rate but low reviewed agreement) are invisible in run-level summary stats, they
only appear when you compare metric M1 vs. metric M2 for the same group G in the table.

**Artifacts are layered evidence.** A run's logged artifacts represent the experiment's
state at multiple levels of granularity. Aggregate summaries, per-group breakdowns,
sample data, evaluation slices. Each artifact reveals a different facet. Read every
relevant artifact the run has logged; do not stop at summary metrics when finer-grained
tables exist. The anomaly almost always lives in a layer below the headline number.

**Weave traces are a required evidence dimension.** If the project has Weave traces,
they are not optional context, they are first-class evidence alongside W&B artifacts.
When runs have collection, inference, or agent-execution traces, query the trace data and
group by model, agent, or task type. Per-model or per-agent success rates are almost never
visible in W&B run summaries — they only appear when you inspect the traces. A single
groupby dimension is not enough: if traces have both a model dimension and a task-type
dimension, query the cross-product (model × task_type).

> **Gate:** Do not advance to Phase 3 until the data has stopped producing new signals.
> Ask yourself: is there any artifact table, trace source, or metric slice you have not
> yet read that could contradict or further localize your current anomaly? If yes, check
> it first. Only when you can answer "no" have you earned a Phase 3 conclusion.
>
> The test for Phase 3 readiness is not "did I analyze the data". It is "can I name the
> specific culprit?" If your best current answer is a general mechanism ("data quality
> issue," "low success rate," "class imbalance") rather than a specific named element
> (which source? which model? which task type?), you have not earned Phase 3 yet.

---

## Phase 3: Synergize with Domain Theory

Connect the named surprise to an established theoretical mechanism. Do not speculate in
a vacuum.

For each candidate explanation, ask: does existing ML theory predict this behavior for
the identified anomaly?

| Anomaly type | Candidate mechanisms to evaluate |
|---|---|
| Metric is flat across all runs | Metric is a derived aggregate (computed at data-prep time, not at run time); shared upstream data artifact; logging error; metric definition mismatch |
| Improvement on primary metric, degradation on secondary | Distribution shift; class imbalance masking; label noise concentrated in tail classes; representation collapse; model overfit to dominant group |
| One source/group degrades the whole | Domain mismatch; systematic annotation error; noisy pseudo-labels; out-of-distribution unlabeled data; contamination |
| Strong baseline, weak fine-tune | Catastrophic forgetting; learning rate too high; insufficient regularization; train/eval distribution gap |

State the mechanism explicitly in plain language: "The flat X metric across all runs
suggests it measures [invariant property], not [per-run outcome]" or "The high-acceptance
/ low-agreement contradiction for element Y is consistent with [theoretical mechanism]."

---

## Phase 4: Design a Falsifiable Probe

Generate a **single-variable experiment** that proves or disproves the mechanism from
Phase 3. This is the recommended next experiment.

**Requirements for a valid probe:**
- Holds all other variables constant: same model, optimizer, seed, training budget,
  labeled dataset, and evaluation protocol
- Is the minimal viable test: a targeted data filter, a controlled ablation, a missing
  evaluation metric, or a single config change
- Has a clear binary outcome: if the mechanism is correct, metric X should change in
  direction Y; if incorrect, it should not

**Valid probe structure:** "Run [minimal experiment] holding [list of fixed variables]
constant. If [mechanism] is correct, we expect [specific measurable outcome]. If [metric]
does not change, we rule out [mechanism] and should instead investigate [alternative]."

**Anti-patterns — do not recommend these as the primary next step:**
- Scaling to a larger model or longer training schedule (not a mechanism test)
- Running a broad hyperparameter sweep (not targeted at the identified anomaly)
- Adding more unlabeled data (does not isolate the cause)
- Switching to a different algorithm class without a controlled comparison

---

## Deliverable

**Every specific number you cite in your final response must appear verbatim in a prior script output.** If you need a per-group breakdown, fetch and print it explicitly before reporting it. Do not estimate or recall values from memory.

Every hypothesis-generation response must end with three things:

1. **The anomaly** — the exact element, metric, and value that violated the prior
2. **The mechanism** — the theoretical explanation grounded in domain knowledge
3. **The probe** — one falsifiable experiment with all other variables held constant

When the anomaly involves a comparison between groups (models, sources, configurations),
present the key metrics side-by-side in a single compact table in your **final written
response** (not only in script terminal output) so all group names and their values are
immediately visible in the same table; do not describe each group in separate paragraphs.
