# Architecture — LLMOps Pipeline

## System Overview

```mermaid
graph TD
    subgraph "Development Loop"
        A["prompts/v1_baseline.txt\nprompts/v2_improved.txt\nprompts/registry.json"] --> PM[prompt_manager.py\nload_prompt version]
        PM --> CB[chain_builder\nRAG chain with versioned prompt]
        EV["eval_datasets/qa_pairs.json"] --> CB
        CB --> RAGAS["RAGAS evaluate()\nGemini-as-judge"]
        RAGAS --> DF[Eval Results DataFrame]
        DF --> MLF[MLflow\nlog params + metrics + artifacts]
        DF --> PM2[prompt_manager\nupdate_prompt_metrics]
        MLF --> DASH["mlflow ui\nhttp://localhost:5000"]
    end

    subgraph "A/B Testing"
        CFA["Variant A config\nprompt_version=v1, k=4"] --> ABT[ab_tester.py\nrun_ab_test]
        CFB["Variant B config\nprompt_version=v2, k=4"] --> ABT
        ABT --> CMP["Comparison Table\nΔ per metric + winner"]
        ABT --> MLF
    end

    subgraph "CI/CD Pipeline — GitHub Actions"
        GIT["git push → main\n(src/ or prompts/ changed)"] --> GHA[".github/workflows/eval_regression.yml"]
        GHA --> PYTEST["pytest tests/ -v"]
        PYTEST --> RC["src/regression_checker.py"]
        RC --> GATE{All metrics\n≥ threshold?}
        GATE -->|Yes| PASS["✅ Build passes\nMerge allowed"]
        GATE -->|No| FAIL["❌ Build fails\nMerge blocked"]
        GHA --> ART["Upload results/\nas build artifact"]
    end
```

## CI Quality Thresholds

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| `faithfulness` | ≥ 0.75 | Below this, the LLM is hallucinating too often to be trusted |
| `answer_relevancy` | ≥ 0.70 | Below this, too many answers miss the actual question |
| `context_recall` | ≥ 0.65 | Below this, retrieval is missing too much relevant information |
| `context_precision` | ≥ 0.65 | Below this, too much noise is being injected into the context |

## Prompt Versioning Model

```
prompts/
├── v1_baseline.txt      ← minimal prompt, no anti-hallucination constraints
├── v2_improved.txt      ← adds ONLY-context constraint, direct answer instruction
└── registry.json        ← maps versions → files → eval scores
                            tracks which version is "current_production"
```

Every prompt change:
1. Creates a new numbered file (`v3_*.txt`)
2. Adds an entry to `registry.json`
3. Runs through the eval pipeline before being deployed
4. Gets its scores written back to `registry.json` via `update_prompt_metrics()`

## MLflow Run Structure

Each `log_eval_run()` call creates one MLflow run with:

```
Run: "v2_tighter_prompt"
├── Params
│   ├── prompt_version: v2
│   ├── model: gemini-2.0-flash
│   ├── chunk_size: 1000
│   └── k: 4
├── Metrics
│   ├── mean_faithfulness: 0.89
│   ├── min_faithfulness: 0.72
│   ├── std_faithfulness: 0.08
│   └── ... (same for all 4 metrics)
├── Artifacts
│   └── v2_tighter_prompt_results.csv
└── Tags
    ├── dataset_size: 25
    └── environment: development
```
