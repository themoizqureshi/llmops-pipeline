"""
Regression checker — the CI/CD quality gate.

This script runs in GitHub Actions on every push to main that touches
src/ or prompts/. Two modes:

  --results <path>   Check a pre-existing eval results CSV (original mode)
  --live             Call the live RAG API, run RAGAS inline, then check thresholds

Live mode requires:
  RAG_API_URL        e.g. http://localhost:8001 or https://your-cloud-run-url
  OPENROUTER_API_KEY or GOOGLE_API_KEY  (for RAGAS judge LLM)

CI exits with code 1 if any metric is below threshold, blocking the merge.

Threshold philosophy:
- Start conservative (0.60) and raise as the system improves
- Set thresholds ~0.05 below your current best score
- Never lower a threshold once established in production
"""

import glob
import logging
import os
import sys

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

THRESHOLDS = {
    "faithfulness": 0.75,
    "answer_relevancy": 0.70,
    "context_recall": 0.65,
    "context_precision": 0.65,
}

# Sample questions used in live CI evaluation
_CI_QA_PAIRS = [
    {
        "question": "What is retrieval-augmented generation?",
        "ground_truth": "RAG combines a retrieval step that fetches relevant documents with a generation step where an LLM produces an answer grounded in those documents.",
    },
    {
        "question": "What are the main components of a RAG pipeline?",
        "ground_truth": "A RAG pipeline has an indexing phase (chunking, embedding, vector store ingestion) and a query phase (embed query, similarity search, context injection, LLM generation).",
    },
    {
        "question": "What is faithfulness in RAG evaluation?",
        "ground_truth": "Faithfulness measures whether the generated answer is grounded in the retrieved context rather than hallucinating information not present in the retrieved documents.",
    },
]


def check_regression(eval_results_path: str) -> bool:
    """Load eval results CSV and check each metric against thresholds."""
    df = pd.read_csv(eval_results_path)

    print("\n" + "=" * 60)
    print("REGRESSION CHECK")
    print(f"Results file: {eval_results_path}")
    print(f"Questions evaluated: {len(df)}")
    print("=" * 60)

    all_pass = True
    for metric, threshold in THRESHOLDS.items():
        if metric not in df.columns:
            logger.warning(f"  ⚠️  {metric}: not found in results (skipping)")
            continue

        actual = df[metric].mean()
        passed = actual >= threshold
        status = "✅ PASS" if passed else "❌ FAIL"
        gap = actual - threshold

        print(f"  {status}  {metric:25s}: {actual:.3f}  (threshold: {threshold:.3f}, gap: {gap:+.3f})")

        if not passed:
            all_pass = False

    print("=" * 60)
    if all_pass:
        print("✅ All quality gates passed. Safe to deploy.")
    else:
        print("❌ Quality regression detected. Check your prompt or code changes.")
        print("   Tip: run the eval locally and inspect LangSmith traces for failing questions.")
    print()

    return all_pass


def run_live_eval(api_url: str, qa_pairs: list[dict]) -> pd.DataFrame:
    """
    Call the live RAG API for each question, build a RAGAS dataset, and score it.

    Returns the results DataFrame (same shape as the CSV-based path).
    """
    import httpx
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
    from langchain_huggingface import HuggingFaceEmbeddings

    logger.info(f"Calling live RAG API at {api_url} for {len(qa_pairs)} questions")

    data: dict[str, list] = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for i, pair in enumerate(qa_pairs):
        q = pair["question"]
        logger.info(f"  [{i+1}/{len(qa_pairs)}] {q[:60]}")
        resp = httpx.post(f"{api_url}/chat", json={"question": q}, timeout=60.0)
        resp.raise_for_status()
        body = resp.json()

        answer = body.get("answer", "")
        sources = body.get("sources", [])
        contexts = [s.get("content", str(s)) if isinstance(s, dict) else str(s) for s in sources]
        if not contexts:
            contexts = [answer]

        data["question"].append(q)
        data["answer"].append(answer)
        data["contexts"].append(contexts)
        data["ground_truth"].append(pair["ground_truth"])

    dataset = Dataset.from_dict(data)

    def _get_judge_llm():
        key = os.getenv("OPENROUTER_API_KEY")
        if key:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model="google/gemini-2.0-flash-001",
                openai_api_key=key,
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=0,
            )
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

    logger.info("Scoring with RAGAS…")
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=_get_judge_llm(),
        embeddings=HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5"),
    )
    return result.to_pandas()


def find_latest_results() -> str:
    """Find the most recently written results CSV."""
    pattern = "results/*.csv"
    files = sorted(glob.glob(pattern))
    if not files:
        logger.error(f"No eval results found matching '{pattern}'. Run evaluation first.")
        sys.exit(1)
    return files[-1]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG regression quality gate")
    parser.add_argument("--results", default=None, help="Path to eval results CSV (default: latest in results/)")
    parser.add_argument("--live", action="store_true", help="Run live evaluation against RAG_API_URL instead of loading CSV")
    args = parser.parse_args()

    if args.live:
        api_url = os.getenv("RAG_API_URL", "").rstrip("/")
        if not api_url:
            logger.error("--live requires RAG_API_URL env var (e.g. http://localhost:8001)")
            sys.exit(1)

        df = run_live_eval(api_url, _CI_QA_PAIRS)

        os.makedirs("results", exist_ok=True)
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"results/live_ci_{ts}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Live eval results saved → {csv_path}")
        passed = check_regression(csv_path)
    else:
        results_path = args.results or find_latest_results()
        logger.info(f"Checking: {results_path}")
        passed = check_regression(results_path)

    sys.exit(0 if passed else 1)
