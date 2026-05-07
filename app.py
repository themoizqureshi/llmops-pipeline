"""
Streamlit UI for the LLMOps Pipeline.

Two modes:
  Demo  — loads pre-computed A/B results (no API calls needed)
  Live  — runs the regression checker against any results CSV

Run with: streamlit run app.py
"""

import json
import os
import glob

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="LLMOps Pipeline", page_icon="🔬", layout="wide")

METRICS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
THRESHOLDS = {
    "faithfulness": 0.75,
    "answer_relevancy": 0.70,
    "context_recall": 0.65,
    "context_precision": 0.65,
}
REGISTRY_PATH = "prompts/registry.json"
SAMPLE_V1 = "results/sample_v1_baseline.csv"
SAMPLE_V2 = "results/sample_v2_improved.csv"


def load_registry() -> dict:
    if not os.path.exists(REGISTRY_PATH):
        return {"versions": [], "current_production": "v1"}
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def load_prompt_file(file_path: str) -> str:
    if os.path.exists(file_path):
        return open(file_path).read()
    return "(prompt file not found)"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Mode")
    mode = st.radio("", ["Demo (pre-computed)", "Run regression check"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Quality Thresholds**")
    for m, t in THRESHOLDS.items():
        st.markdown(f"**{m.replace('_', ' ').title()}** ≥ {t}")

    st.markdown("---")
    st.markdown("**Stack**")
    st.markdown("MLflow · RAGAS · GitHub Actions")
    st.markdown("Gemini 2.0 Flash via OpenRouter")


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔬 LLMOps Pipeline")
st.caption("Prompt versioning · A/B testing · MLflow experiment tracking · CI/CD regression gate")

tab1, tab2, tab3 = st.tabs(["Prompt Registry", "A/B Comparison", "Regression Gate"])


# ── Tab 1: Prompt Registry ────────────────────────────────────────────────────
with tab1:
    registry = load_registry()
    versions = registry.get("versions", [])
    prod = registry.get("current_production", "v1")

    st.subheader("Prompt Version Registry")
    st.caption(f"Production version: **{prod}** · Stored in `prompts/registry.json`")

    if not versions:
        st.warning("No prompt versions found. Check `prompts/registry.json`.")
    else:
        for v in versions:
            is_prod = v["version"] == prod
            badge = "🟢 **PRODUCTION**" if is_prod else "⚪ archived"
            with st.expander(f"v{v['version'].upper()} — {v['description']}  {badge}"):
                col_a, col_b = st.columns([1, 1])
                with col_a:
                    st.markdown("**Prompt file:**")
                    st.code(load_prompt_file(v["file"]), language="text")
                with col_b:
                    st.markdown("**Evaluation metrics:**")
                    metrics = v.get("metrics") or {}
                    for m in METRICS:
                        val = metrics.get(m)
                        if val is not None:
                            threshold = THRESHOLDS[m]
                            status = "✅" if val >= threshold else "❌"
                            st.markdown(f"{status} {m.replace('_', ' ').title()}: **{val:.3f}** (≥ {threshold})")
                        else:
                            st.markdown(f"⬜ {m.replace('_', ' ').title()}: not evaluated")


# ── Tab 2: A/B Comparison ─────────────────────────────────────────────────────
with tab2:
    if not os.path.exists(SAMPLE_V1) or not os.path.exists(SAMPLE_V2):
        st.warning("Pre-computed results not found at `results/sample_*.csv`. Run an evaluation first.")
    else:
        v1 = pd.read_csv(SAMPLE_V1)
        v2 = pd.read_csv(SAMPLE_V2)

        st.subheader("A/B Test — v1 Baseline vs. v2 Strict Prompt")
        st.caption(
            "Single variable changed: system prompt strictness. "
            "All other params identical (model: gemini-2.0-flash, chunk_size: 1000, k: 4)."
        )

        cols = st.columns(4)
        for i, metric in enumerate(METRICS):
            v1_mean = v1[metric].mean()
            v2_mean = v2[metric].mean()
            delta = v2_mean - v1_mean
            winner = "B" if delta > 0.01 else ("A" if delta < -0.01 else "tie")
            cols[i].metric(
                label=metric.replace("_", " ").title(),
                value=f"{v2_mean:.3f}",
                delta=f"{delta:+.3f} vs v1",
                delta_color="normal" if delta >= 0 else "inverse",
            )

        st.markdown("---")
        chart_data = pd.DataFrame(
            {
                "v1 Baseline": [v1[m].mean() for m in METRICS],
                "v2 Strict": [v2[m].mean() for m in METRICS],
            },
            index=[m.replace("_", " ").title() for m in METRICS],
        )
        st.bar_chart(chart_data, color=["#94a3b8", "#3b82f6"])

        st.markdown("---")
        st.markdown("**Verdict**")
        b_wins = sum(1 for m in METRICS if v2[m].mean() - v1[m].mean() > 0.01)
        a_wins = sum(1 for m in METRICS if v1[m].mean() - v2[m].mean() > 0.01)
        if b_wins > a_wins:
            st.success(f"v2 wins {b_wins}/{len(METRICS)} metrics. Promote to production.")
        elif a_wins > b_wins:
            st.error(f"v1 still stronger on {a_wins}/{len(METRICS)} metrics. Do not promote v2.")
        else:
            st.info("Tie — insufficient evidence to promote. Run more experiments.")


# ── Tab 3: Regression Gate ────────────────────────────────────────────────────
with tab3:
    st.subheader("Regression Gate")

    if mode == "Demo (pre-computed)":
        st.info("Running regression check against pre-computed v2 results.", icon="ℹ️")
        results_path = SAMPLE_V2
    else:
        all_csvs = sorted(glob.glob("results/*.csv"))
        if not all_csvs:
            st.warning("No results CSVs found in `results/`. Run an evaluation first.")
            st.stop()
        results_path = st.selectbox("Select results CSV", all_csvs, index=len(all_csvs) - 1)

    if results_path and os.path.exists(results_path):
        df = pd.read_csv(results_path)
        st.caption(f"File: `{results_path}` · {len(df)} questions evaluated")

        all_pass = True
        for metric in METRICS:
            if metric not in df.columns:
                st.warning(f"⚠️ {metric} not found in results (skipping)")
                continue
            actual = df[metric].mean()
            threshold = THRESHOLDS[metric]
            passed = actual >= threshold
            if not passed:
                all_pass = False
            status = "✅ PASS" if passed else "❌ FAIL"
            gap = actual - threshold
            col_a, col_b, col_c = st.columns([2, 1, 1])
            col_a.markdown(f"{status} **{metric.replace('_', ' ').title()}**")
            col_b.markdown(f"`{actual:.3f}`")
            col_c.markdown(f"threshold ≥ {threshold} &nbsp; `{gap:+.3f}`")

        st.markdown("")
        if all_pass:
            st.success("✅ All quality gates passed. Safe to deploy.")
        else:
            st.error(
                "❌ Quality regression detected. "
                "Check failing questions, inspect LangSmith traces, and review recent prompt/code changes."
            )

        st.markdown("---")
        st.markdown("**What runs this in CI:**")
        st.code("python src/regression_checker.py  # exits 1 on failure → fails GitHub Actions build", language="bash")
