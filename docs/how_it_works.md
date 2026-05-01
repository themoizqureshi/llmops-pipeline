# How It Works — LLMOps Pipeline

## The Core Insight

Projects 1–4 build AI systems. Project 5 is about *operating* them in production. The distinction matters:

- **Building**: get a working RAG system with good eval scores on your dev machine
- **Operating**: ensure those scores stay high after every code change, catch regressions before they reach users, understand *why* scores changed, and enable safe rollbacks

This is what LLMOps means: applying software engineering discipline (versioning, CI/CD, experiment tracking) to the uniquely messy problems of LLM systems (non-deterministic outputs, prompt-as-code, LLM-as-judge evaluation).

---

## Component 1: Prompt Versioning (`src/prompt_manager.py`)

```
prompts/
├── v1_baseline.txt       ← "Answer the question: {question}"
├── v2_improved.txt       ← "Only use context. Answer directly. No general knowledge."
└── registry.json         ← {version: "v2", file: "...", metrics: {faithfulness: 0.89}}
```

The registry pattern solves a real operational problem: without it, you have no way to answer "which prompt produced that 0.89 faithfulness score three weeks ago?" With it, every eval score is permanently linked to the exact prompt that produced it.

```python
# Load any version by name
prompt_template = load_prompt("v2")

# After running eval, write scores back
update_prompt_metrics("v2", {"faithfulness": 0.89, "answer_relevancy": 0.81})

# Know what's in production
current = get_current_production_version()  # → "v1"
```

---

## Component 2: MLflow Experiment Tracking (`src/experiment_tracker.py`)

MLflow stores a queryable history of every eval run:

```python
with mlflow.start_run(run_name="v2_tighter_prompt"):
    mlflow.log_params({"prompt_version": "v2", "model": "gemini-2.0-flash", "k": 4})
    mlflow.log_metric("mean_faithfulness", 0.89)
    mlflow.log_metric("min_faithfulness", 0.72)
    mlflow.log_artifact("results/v2_results.csv")
```

Three months from now: open `mlflow ui`, sort by `mean_faithfulness`, find the run where it first dropped below 0.80, click it, see the params — and you know exactly what changed.

**Why track min and std alongside mean?** A mean of 0.89 could hide a min of 0.20 on one question. High std means the system is inconsistent — good for some questions, bad for others. You want high mean AND high floor.

---

## Component 3: A/B Testing (`src/ab_tester.py`)

The cardinal rule: change ONE variable per test.

```python
results_a, results_b, comparison = run_ab_test(
    variant_a_config={"prompt_version": "v1", "k": 4},  # baseline
    variant_b_config={"prompt_version": "v2", "k": 4},  # only prompt changed
    qa_pairs=qa_pairs,
    chain_builder_fn=build_rag_chain,
    evaluator_fn=run_ragas_eval,
)
# Output:
# 🏆 faithfulness        : A=0.72 | B=0.89 | Δ=+0.170 (+23.6%)
#    answer_relevancy     : A=0.68 | B=0.81 | Δ=+0.130 (+19.1%)
# ➖ context_recall       : A=0.74 | B=0.75 | Δ=+0.010 (+1.4%)
# ❌ context_precision    : A=0.80 | B=0.71 | Δ=-0.090 (-11.3%)
```

This confirms: the prompt change improved faithfulness (+23.6%) as intended, but slightly hurt context_precision (-11.3%). Why? The stricter "only use context" instruction may be causing the LLM to rely on fewer chunks, effectively reducing which chunks it "uses" for the answer — even though they were all retrieved. Investigation needed.

---

## Component 4: The CI/CD Regression Gate

```
.github/workflows/eval_regression.yml
├── Trigger: push to main touching src/ or prompts/
├── Step 1: pip install -r requirements.txt
├── Step 2: pytest tests/ -v  (unit tests must pass first)
├── Step 3: python src/regression_checker.py
│           ↓ loads latest results/*.csv
│           ↓ checks mean(faithfulness) ≥ 0.75, etc.
│           ↓ exits 0 (pass) or 1 (fail)
└── Step 4: upload results/ as artifact (always, even on failure)
```

**Why upload artifacts even on failure?** When regression_checker fails, you need the CSV to debug which questions failed and why. If artifacts were only uploaded on success, you'd have nothing to examine after a failure.

**Why path-based triggers?** A README update or a test refactor has zero chance of affecting eval scores. Running RAGAS for every commit would cost API quota and slow down CI. The `paths:` filter ensures only changes to `src/` (code) and `prompts/` (prompts) trigger the quality gate.

---

## The Full Workflow

```
Developer makes a prompt change:
  Edit prompts/v3_concise.txt
  Add v3 entry to prompts/registry.json
  git push

GitHub Actions fires:
  1. pytest tests/               → passes (no unit test failures)
  2. regression_checker.py       → loads results/latest.csv
                                 → mean_faithfulness = 0.61 < 0.75
                                 → exits 1 → CI fails

Developer investigates:
  Download eval artifacts from GitHub Actions
  Open LangSmith traces for low-scoring questions
  See: v3 prompt is too concise, LLM ignoring context chunks

Developer fixes:
  Update v3 prompt
  Run local eval: python run_eval.py --prompt-version v3
  Confirm scores above thresholds
  git push → CI passes → PR mergeable
```

This is the discipline that separates experimental AI engineering from production AI engineering.
