# Interview Prep — LLMOps Pipeline

> Senior AI roles (₹40–50 LPA) don't just ask "can you build a RAG system?" They ask "how do you operate it in production?" This project is your answer.

---

## Core Concept Questions

### Q: What is LLMOps and how is it different from MLOps?

> "MLOps handles the traditional ML lifecycle: data versioning, model training, experiment tracking with numeric metrics, deployment, and drift monitoring. LLMOps is a superset that adds several LLM-specific concerns:
>
> **Prompt versioning**: prompts are code. A change to a system prompt is a code deployment that affects system behavior — it needs version control, testing, and rollback capability. Traditional MLOps has no concept of this.
>
> **Non-deterministic quality evaluation**: you can't use accuracy as your metric because LLM outputs are free-form text. You need RAGAS (LLM-as-judge) for quality, not just pass/fail unit tests.
>
> **Hallucination monitoring**: traditional models produce wrong predictions; LLMs confidently produce wrong text. You need faithfulness-specific checks, not just error rate monitoring.
>
> **Cost as a metric**: token cost is a first-class concern. Every eval run, every production query costs money. MLflow tracks latency and cost alongside quality scores.
>
> The toolchain is also different: MLflow for experiment tracking, LangSmith for trace observability, RAGAS for evaluation — on top of the traditional CI/CD stack."

---

### Q: Walk me through your CI/CD pipeline for LLM quality.

> "On every push to main that touches `src/` or `prompts/`, GitHub Actions runs two steps: unit tests (pytest), then the regression evaluator.
>
> The regression evaluator loads the most recent eval results CSV from the `results/` directory. It computes the mean of each RAGAS metric and compares against hard thresholds — faithfulness ≥ 0.75, answer_relevancy ≥ 0.70, context_recall ≥ 0.65, context_precision ≥ 0.65. If any metric is below threshold, the script exits with code 1, which fails the GitHub Actions build and blocks the PR merge.
>
> The results are uploaded as build artifacts even when the check fails — so I can download the per-question breakdown and use LangSmith traces to see exactly which questions caused the regression and why.
>
> The path-based trigger (`paths: ['src/**', 'prompts/**']`) is intentional: a README change or a test refactor shouldn't trigger an expensive RAGAS eval run. Only changes that could affect answer quality trigger the gate."

---

### Q: How do you version prompts and why does it matter?

> "I treat prompts identically to code. Every prompt version has its own file in `prompts/` (e.g., `v1_baseline.txt`, `v2_improved.txt`), a registry entry in `registry.json` that maps the version to the file and tracks its measured eval scores, and a 'current_production' field in the registry that tells the system which version is live.
>
> Why it matters: the system prompt is often the highest-leverage variable in a RAG system. Changing 'Answer the question' to 'Answer the question directly before adding context' can improve answer_relevancy by 15 points. If you don't version it, you can't reproduce past results, you can't roll back when a prompt change tanks faithfulness, and you can't reason about what caused a production quality drop.
>
> In `prompt_manager.py`, `load_prompt('v2')` reads from the registry and loads the file — this is a single line change to switch the production prompt. After running eval on the new version, `update_prompt_metrics('v2', scores)` writes the measured scores back to the registry so it's always up to date."

---

### Q: How do you set regression thresholds and what happens when scores vary between runs?

> "I set thresholds at approximately 0.05 below the current best measured score. If my best faithfulness is 0.89, I set the threshold at 0.75 — not 0.88. This gives a buffer for natural run-to-run variance in LLM-as-judge scoring.
>
> Why does variance exist? RAGAS uses Gemini to judge each answer. The same answer evaluated twice might score 0.87 and 0.91 due to non-determinism in the judge. If I set the threshold at 0.88, a genuinely good prompt might fail CI 30% of the time just due to this variance — creating false positives that erode trust in the gate.
>
> A 0.05 buffer absorbs normal variance while still catching genuine regressions (a bad prompt change will typically drop faithfulness by 0.10+, not 0.03).
>
> For production systems with higher stakes, I'd run each eval sample 3 times and take the mean to reduce variance, then set tighter thresholds."

---

### Q: How does MLflow fit into this pipeline?

> "MLflow gives me a queryable history of every experiment. Each eval run creates one MLflow run with three types of data: parameters (what I changed — model, prompt_version, chunk_size, k), metrics (what resulted — mean/min/std of all four RAGAS metrics), and artifacts (the full results CSV).
>
> The practical benefit: when a score drops in CI three months from now, I can open the MLflow UI, sort by faithfulness, and immediately see which prompt version first dropped below 0.80 and which parameter change correlated with it. Without MLflow, I'd be guessing.
>
> In the MLflow UI, I can also do visual A/B comparisons — select two runs and see a side-by-side metric comparison. This is more intuitive than reading CSV diffs.
>
> For production at scale, I'd use a hosted MLflow server (Databricks or a self-hosted instance) rather than local `./mlruns/`. The code is identical — just change `MLFLOW_TRACKING_URI`."

---

### Q: What's the difference between this eval pipeline and the one in Project 2?

> "Project 2 is the standalone evaluation tool — it answers 'how good is my current RAG system?' with RAGAS scores and a before/after comparison. It's a developer tool for iterative improvement.
>
> Project 5 is the production gate — it integrates that evaluation into a CI/CD pipeline that blocks deployments automatically. It adds prompt versioning (registry.json), MLflow experiment tracking across all runs (not just two), an A/B testing harness for controlled experiments, and the GitHub Actions workflow that runs automatically on every relevant push.
>
> The analogy: Project 2 is like running a test suite manually. Project 5 is like setting up CI so the test suite runs automatically on every commit and blocks merges when it fails. Same underlying evaluation, completely different operational posture."

---

## Connecting to Your Production Experience

> "At Speridian, I achieved ~99% field-level accuracy on mortgage document extraction. That wasn't achieved by checking manually — I built scripts that compared extracted fields against manually verified ground truth and tracked metrics over time. This project is that same discipline applied to open-ended LLM outputs: instead of exact field matching, I need RAGAS because answers are free-form. But the principle — measure, find failures, fix, measure again — is identical."

> "The prompt versioning pattern here maps directly to feature flag management in traditional software. At Speridian, feature flags were deployed via configuration changes, not code deployments — same philosophy as prompt versioning. The registry.json is the feature flag store; `current_production: 'v2'` is the flag state; rolling back means changing it back to `'v1'`."
