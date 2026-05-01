# LLMOps Pipeline — MLflow + RAGAS + GitHub Actions CI/CD

> Prompt versioning, A/B testing, MLflow experiment tracking, and automated regression gates on every push. The difference between a demo and a production AI system — in one repo.

![Python](https://img.shields.io/badge/python-3.11-blue)
![MLflow](https://img.shields.io/badge/MLflow-2.18-blue)
![RAGAS](https://img.shields.io/badge/RAGAS-0.2.6-purple)
![CI](https://img.shields.io/badge/CI-GitHub_Actions-green)

---

## Skills Demonstrated

| Category | Technologies / Concepts |
|----------|------------------------|
| **LLMOps** | Prompt versioning, experiment tracking, quality gates, regression detection |
| **Experiment Tracking** | MLflow: log_params, log_metric, log_artifact, run comparison in UI |
| **A/B Testing** | Single-variable experiments, controlled comparison, statistical interpretation |
| **CI/CD for AI** | GitHub Actions, path-based triggers, `sys.exit(1)` quality gates, artifact upload |
| **Prompt Engineering** | Registry pattern, version → file → metric mapping, rollback capability |
| **Evaluation** | RAGAS with Gemini-as-judge (reused from Project 2) |
| **Software Engineering** | pytest with `tmp_path`, clean threshold logic, separation of concerns |

---

## What This Builds

**The Problem:** You've built a great RAG system (Projects 1–3). Now answer: "How do you know when a code or prompt change breaks it before it reaches users?" Without systematic tracking, a prompt tweak that tanks faithfulness goes undetected until users complain.

**The Solution:** A complete LLMOps pipeline:
1. **Prompt versioning** — prompts are code, in files, with a registry
2. **MLflow tracking** — every eval run is logged with all parameters and metrics
3. **A/B testing harness** — controlled single-variable experiments
4. **CI/CD quality gate** — GitHub Actions fails the build if any metric drops below threshold

**The Outcome:** Every push that touches `src/` or `prompts/` runs an automated quality check. Bad changes are caught before merge, not after deploy.

---

## Architecture

```mermaid
graph TD
    subgraph "Development"
        P["prompts/v1_baseline.txt\nprompts/v2_improved.txt"] --> PM[prompt_manager.py]
        PM --> RG["registry.json\n{version, file, metrics}"]
        PM --> CH[RAG Chain Builder]
        EV["eval_datasets/qa_pairs.json"] --> CH
        CH --> RA["RAGAS evaluate()\nGemini-as-judge"]
        RA --> DF[Results DataFrame]
        DF --> MLF["MLflow\nparams + metrics + CSV artifact"]
        MLF --> UI["mlflow ui\nlocalhost:5000"]
    end

    subgraph "CI/CD"
        GIT["git push → main\n(src/ or prompts/ changed)"] --> GHA[GitHub Actions]
        GHA --> PT[pytest tests/]
        PT --> RC["regression_checker.py\ncheck mean ≥ threshold"]
        RC --> |"exit 0"| PASS["✅ Merge allowed"]
        RC --> |"exit 1"| FAIL["❌ Build failed — merge blocked"]
    end
```

---

## Key Engineering Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| **Prompt storage** | Files in `prompts/` + `registry.json` | Git-versioned, auditable, rollback = `git checkout prompts/v1_baseline.txt` |
| **Threshold buffer** | 0.05 below current best | Absorbs LLM-as-judge variance (~±0.03) while catching real regressions (typically >0.10 drop) |
| **Path-based CI trigger** | `paths: ['src/**', 'prompts/**']` | README and test refactors don't burn API quota on unnecessary RAGAS runs |
| **Artifact upload on failure** | `if: always()` | Failed CI runs produce the most valuable debugging data — you need the CSV |
| **MLflow local storage** | `./mlruns` | Zero setup for a portfolio project; change `MLFLOW_TRACKING_URI` for production |
| **min + std alongside mean** | Logged for all metrics | Mean of 0.89 can hide a min of 0.20; high std signals inconsistency |
| **Single-variable A/B** | Enforced by convention | Two simultaneous changes = uninterpretable results; documented in `ab_tester.py` |

---

## Tech Stack

| Component | Technology | Version | Why |
|-----------|-----------|---------|-----|
| Experiment Tracking | MLflow | 2.18.0 | Industry standard; UI comparison, artifact storage, query API |
| Evaluation | RAGAS | 0.2.6 | Reuses Project 2 eval framework; consistent metrics across projects |
| Judge LLM | Gemini 2.0 Flash | `langchain-google-genai` | Free, consistent with Project 2 |
| CI/CD | GitHub Actions | — | Free for public repos; integrates with PR merge protection |
| Quality Gate | Python `sys.exit(1)` | — | Simplest contract with CI: non-zero exit = failure |
| Prompt Registry | JSON file | — | Human-readable, git-diffs cleanly, no database needed at this scale |

---

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/llmops-pipeline
cd llmops-pipeline

cp .env.example .env
# Add GOOGLE_API_KEY (same as Projects 1-2)

uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Start MLflow UI (open in browser)
mlflow ui
# → http://localhost:5000
```

**Running an eval and logging to MLflow:**
```python
from src.experiment_tracker import setup_mlflow, log_eval_run
from src.prompt_manager import load_prompt

setup_mlflow()
# ... build your RAG chain with load_prompt("v2") ...
# ... run RAGAS eval ...
run_id = log_eval_run(eval_df, params={"prompt_version": "v2", "k": 4}, run_name="v2_eval")
```

**Running the regression check locally:**
```bash
# Assumes results/some_eval.csv exists
python src/regression_checker.py
# → prints PASS/FAIL per metric, exits 0 or 1
```

## Running Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
llmops-pipeline/
├── .github/
│   └── workflows/
│       └── eval_regression.yml    # CI pipeline: pytest → regression_checker
├── src/
│   ├── experiment_tracker.py      # setup_mlflow(), log_eval_run(), compare_runs_in_mlflow()
│   ├── ab_tester.py               # run_ab_test() — controlled single-variable comparisons
│   ├── regression_checker.py      # check_regression() — CI quality gate, exits 1 on failure
│   └── prompt_manager.py          # load_prompt(), update_prompt_metrics(), registry CRUD
├── prompts/
│   ├── v1_baseline.txt            # Minimal prompt — baseline for A/B testing
│   ├── v2_improved.txt            # Improved prompt with anti-hallucination constraints
│   └── registry.json              # Version → file → metrics mapping
├── eval_datasets/
│   └── qa_pairs.json              # Q&A eval set (fill in for your PDF)
├── results/                       # CSVs saved here (gitignored except .gitkeep)
└── docs/
    ├── architecture.md            # MLflow run structure, CI flow, threshold table
    ├── how_it_works.md            # Prompt versioning deep-dive, A/B testing rationale
    └── interview_prep.md          # Q&A: LLMOps vs MLOps, CI pipeline, threshold design, prompt versioning
```

---

## CI Setup (GitHub Secrets Required)

To activate the CI pipeline, add these secrets in your repo's Settings → Secrets:

| Secret | Value |
|--------|-------|
| `GOOGLE_API_KEY` | Your Google AI Studio key |
| `LANGCHAIN_API_KEY` | Your LangSmith key (optional) |

The workflow triggers automatically on push to `main` that touches `src/` or `prompts/`.

---

## Production Considerations

| Concern | Current State | Production Approach |
|---------|--------------|---------------------|
| **MLflow storage** | Local `./mlruns/` (lost on CI runner) | Hosted MLflow server or Databricks; set `MLFLOW_TRACKING_URI` |
| **Eval cost in CI** | Full RAGAS on every push | Use a smaller held-out eval set (10 questions) for CI; full eval runs nightly |
| **Judge variance** | Single RAGAS run per eval | Run 3× and take mean for tighter threshold enforcement |
| **Prompt registry** | JSON file | At scale: database with audit log, approval workflow for prompt changes |
| **Notification on failure** | GitHub email only | Slack webhook on CI failure; include which metric failed and by how much |

---

## Lessons Learned

- *Fill in after building. Suggested prompts:*
  - *Did the CI gate ever block a change that was actually fine? How did you tune the threshold?*
  - *What was the most useful thing the MLflow UI showed you?*
  - *What would you add to the prompt registry for a real production system?*

---

## Resume Bullet Points

> **Designed LLMOps CI/CD pipeline** with GitHub Actions: automated RAGAS evaluation on every push to `src/` or `prompts/`, threshold-based quality gates (`sys.exit(1)` blocks merges on regression), and artifact upload for post-failure debugging.

> **Implemented prompt versioning system** with file-based version registry (`prompts/registry.json`), enabling prompt rollback, cross-version metric comparison, and audit trail of which prompt produced each eval score.

> **Built A/B testing harness** with MLflow experiment tracking — logging parameters (prompt_version, chunk_size, k), metrics (mean/min/std of RAGAS scores), and CSV artifacts per run for queryable experiment history.

---

*Part of the [AI Engineer Portfolio](https://github.com/YOUR_USERNAME) — Project 5 of 5.*  
*Previous: [Project 4 — Multi-Agent LangGraph](https://github.com/YOUR_USERNAME/multi-agent-langgraph)*  
*See [PORTFOLIO.md](../PORTFOLIO.md) for the full story arc across all 5 projects.*
