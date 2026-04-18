"""
evaluate.py — RAGAS evaluation for the PubMed cardiology RAG pipeline.

Run directly:
    uv run python evaluate.py

Or call run_ragas_eval() from main.py via the /evaluate endpoint.
"""

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from datasets import Dataset
from dotenv import load_dotenv
import pandas as pd
import json
import os

load_dotenv()

# --- Default cardiology test cases ---
# Add ground_truth to enable context_recall metric.
# Without ground_truth, only faithfulness, answer_relevancy,
# and context_precision are scored.

DEFAULT_TEST_CASES = [
    {
        "question": "What are the first-line pharmacological treatments for heart failure with reduced ejection fraction?",
        "ground_truth": "First-line treatments for HFrEF include ACE inhibitors or ARBs, beta-blockers, and mineralocorticoid receptor antagonists. SGLT2 inhibitors are also now recommended."
    },
    {
        "question": "What is the role of BNP and NT-proBNP in heart failure diagnosis?",
        "ground_truth": "BNP and NT-proBNP are natriuretic peptides used as biomarkers for heart failure diagnosis and prognosis. Elevated levels indicate increased ventricular wall stress."
    },
    {
        "question": "When is cardiac resynchronization therapy indicated in heart failure patients?",
        "ground_truth": "CRT is indicated in patients with symptomatic heart failure, LVEF 35% or less, and QRS duration of 130ms or more with left bundle branch block morphology despite optimal medical therapy."
    },
    {
        "question": "What are the differences between HFpEF and HFrEF in terms of treatment?",
        "ground_truth": "HFrEF has proven pharmacological therapies including ACE inhibitors, beta-blockers, and MRAs. HFpEF treatment focuses on symptom management and comorbidity treatment as fewer disease-modifying therapies exist."
    },
    {
        "question": "What factors predict 30-day hospital readmission in heart failure patients?",
        "ground_truth": "Predictors of 30-day readmission include elevated BNP at discharge, renal dysfunction, prior hospitalizations, poor medication adherence, and lack of early follow-up care."
    },
]


def run_ragas_eval(
    rag,
    test_cases: list[dict],
    use_ground_truth: bool = True
) -> pd.DataFrame:
    """
    Run RAGAS evaluation on a list of test cases.

    Args:
        rag:              PubMedRAGChain instance from app_state
        test_cases:       list of dicts with 'question' and optional 'ground_truth'
        use_ground_truth: include context_recall metric (requires ground_truth)

    Returns:
        DataFrame with per-question scores and aggregate means
    """
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", temperature=0))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print(f"Running evaluation on {len(test_cases)} test cases...")

    for i, case in enumerate(test_cases):
        question = case["question"]
        print(f"  [{i+1}/{len(test_cases)}] {question[:60]}...")

        # Run RAG pipeline with empty history (evaluation is stateless)
        result = rag.query(question, history=[])
        answer = result["answer"]
        docs = result["sources"]

        questions.append(question)
        answers.append(answer)
        contexts.append([doc.page_content for doc in docs])
        ground_truths.append(case.get("ground_truth", ""))

    # Build dataset in the format RAGAS expects
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }

    dataset = Dataset.from_dict(data)

    # Select metrics
    metrics = [faithfulness, answer_relevancy, context_precision]
    if use_ground_truth and any(gt for gt in ground_truths):
        metrics.append(context_recall)

    print("\nScoring with RAGAS...")
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
    )

    df = result.to_pandas()
    return df


KNOWN_METRIC_COLS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]


def get_metric_cols(df: pd.DataFrame) -> list[str]:
    """Return only the numeric RAGAS score columns present in the DataFrame."""
    return [c for c in KNOWN_METRIC_COLS if c in df.columns]


def print_results(df: pd.DataFrame):
    """Pretty print evaluation results to terminal."""
    print("\n" + "="*60)
    print("RAGAS EVALUATION RESULTS")
    print("="*60)

    metric_cols = get_metric_cols(df)

    # Aggregate scores
    print("\nAggregate Scores:")
    for col in metric_cols:
        score = df[col].mean()
        rating = (
            "excellent"        if score >= 0.9 else
            "good"             if score >= 0.7 else
            "needs improvement" if score >= 0.5 else
            "poor"
        )
        print(f"  {col:<25} {score:.3f}  ({rating})")

    # Per-question scores
    print("\nPer-Question Scores:")
    print("-"*60)

    # RAGAS versions use different column names for the question
    question_col = next(
        (c for c in ["user_input", "question", "query"] if c in df.columns),
        None
    )

    for i, (_, row) in enumerate(df.iterrows()):
        q_text = row[question_col][:70] if question_col else f"Question {i+1}"
        print(f"\nQ: {q_text}...")
        for col in metric_cols:
            print(f"  {col:<25} {row[col]:.3f}")

    print("\n" + "="*60)


def save_results(df: pd.DataFrame, path: str = "./eval_results.json"):
    """Save results to JSON for later analysis."""
    metric_cols = get_metric_cols(df)
    question_col = next(
        (c for c in ["user_input", "question", "query"] if c in df.columns),
        None
    )
    cols_to_save = ([question_col] if question_col else []) + metric_cols
    results = {
        "aggregate": {
            col: round(float(df[col].mean()), 3)
            for col in metric_cols
        },
        "per_question": df[cols_to_save].to_dict(orient="records")
    }
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {path}")


# --- Run directly from terminal ---
if __name__ == "__main__":
    from main import app_state, build_rag, load_from_chroma, CHROMA_PATH
    import os

    if not os.path.exists(CHROMA_PATH) or not os.listdir(CHROMA_PATH):
        print("No ChromaDB found. Run seed.py first.")
        exit(1)

    print("Loading knowledge base from ChromaDB...")
    chunks = load_from_chroma()
    rag = build_rag(chunks)
    print(f"Loaded {len(chunks)} chunks.")

    df = run_ragas_eval(rag, DEFAULT_TEST_CASES, use_ground_truth=True)
    print_results(df)
    save_results(df)