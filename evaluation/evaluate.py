import sys
from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

# Allow imports from src/
ROOT_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT_DIR)
)

from src.intent_classifier import IntentClassifier
from src.retrieval import SemanticRetriever
from src.escalation import decide_escalation
from src.generation import ResponseGenerator
from src.config import EMBEDDING_MODEL, TOP_K

from evaluation.baselines import (
    MajorityClassBaseline,
    TfidfBaseline
)

from evaluation.judge import LLMJudge


GOLDEN_PATH = (
    ROOT_DIR
    / "evaluation"
    / "golden_set.csv"
)

BRAND_DATA_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "brand_data.csv"
)

HUMAN_SCORES_PATH = (
    ROOT_DIR
    / "evaluation"
    / "human_judge_scores.csv"
)


def print_metric_block(
    name,
    y_true,
    y_pred
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    )

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"Macro F1 : {f1:.4f}"
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "macro_f1": f1
    }


def evaluate():

    if not GOLDEN_PATH.exists():

        print(
            "golden_set.csv not found."
        )

        print(
            "Create the golden set before evaluation."
        )

        return

    if not BRAND_DATA_PATH.exists():

        print(
            "brand_data.csv not found."
        )

        print(
            "Run prepare_brand_data.py first."
        )

        return

    golden = pd.read_csv(
        GOLDEN_PATH
    )

    brand_data = pd.read_csv(
        BRAND_DATA_PATH
    )

    required_columns = {
        "customer_text",
        "intent"
    }

    missing = (
        required_columns
        - set(golden.columns)
    )

    if missing:

        print(
            f"Golden set is missing columns: {missing}"
        )

        return

    print(
        f"Golden examples: {len(golden)}"
    )

    print(
        f"Historical examples: {len(brand_data)}"
    )

   
    labelled = brand_data[
        brand_data["intent"].notna()
    ].copy()

    if len(labelled) < 20:

        print()
        print(
            "Not enough labelled training examples."
        )

        print(
            "Add intent labels to brand_data.csv "
            "before running evaluation."
        )

        return

   

    majority = MajorityClassBaseline()

    majority.fit(
        labelled["intent"]
    )

    majority_predictions = majority.predict(
        golden["customer_text"]
    )

    print_metric_block(
        "Majority-Class Baseline",
        golden["intent"],
        majority_predictions
    )

    

    tfidf = TfidfBaseline()

    tfidf.fit(
        labelled["customer_text"],
        labelled["intent"]
    )

    tfidf_predictions = tfidf.predict(
        golden["customer_text"]
    )

    print_metric_block(
        "TF-IDF + Logistic Regression",
        golden["intent"],
        tfidf_predictions
    )

    

    classifier = IntentClassifier()

    classifier.fit(
        labelled["customer_text"],
        labelled["intent"]
    )

    predictions = classifier.predict(
        golden["customer_text"]
    )

    predicted_intents = [
        item.intent
        for item in predictions
    ]

    intent_confidences = [
        item.confidence
        for item in predictions
    ]

    print_metric_block(
        "Main Intent Classifier",
        golden["intent"],
        predicted_intents
    )

    

    retriever = SemanticRetriever(
        EMBEDDING_MODEL
    )

    retriever.build(
        labelled["customer_text"]
        .tolist()
    )

    
    escalation_results = []

    for index, row in golden.iterrows():

        retrieved = retriever.search(
            row["customer_text"],
            TOP_K
        )

        best_score = (
            retrieved[0]["score"]
            if retrieved
            else 0.0
        )

        decision = decide_escalation(
            message=row["customer_text"],
            intent_confidence=
                intent_confidences[index],
            retrieval_score=best_score
        )

        escalation_results.append(
            decision
        )

    golden["predicted_intent"] = (
        predicted_intents
    )

    golden["intent_confidence"] = (
        intent_confidences
    )

    golden["escalation"] = [
        item["decision"]
        for item in escalation_results
    ]

    golden["escalation_reason"] = [
        item["reason"]
        for item in escalation_results
    ]

  

    print()
    print(
        "Generating responses..."
    )

    try:

        generator = ResponseGenerator()

    except ValueError as error:

        print(error)

        print(
            "Skipping LLM response generation."
        )

        golden.to_csv(
            GOLDEN_PATH,
            index=False
        )

        return

    generated_responses = []
    retrieved_evidence = []

    for index, row in golden.iterrows():

        retrieved = retriever.search(
            row["customer_text"],
            TOP_K
        )

        evidence = [
            item["text"]
            for item in retrieved
        ]

        retrieved_evidence.append(
            evidence
        )

        escalation = escalation_results[index]

        if (
            escalation["decision"]
            == "AUTO-HANDLE"
        ):

            response = generator.generate(
                customer_message=
                    row["customer_text"],
                intent=
                    predicted_intents[index],
                retrieved_examples=
                    retrieved
            )

        else:

            response = (
                "This request requires "
                "human review."
            )

        generated_responses.append(
            response
        )

    golden["generated_response"] = (
        generated_responses
    )

   
    golden.to_csv(
        GOLDEN_PATH,
        index=False
    )

   

    run_judge = input(
        "\nRun LLM judge on golden set? "
        "(y/n): "
    ).strip().lower()

    if run_judge != "y":

        print(
            "Evaluation complete."
        )

        return

    judge = LLMJudge()

    judge_rows = []

    for index, row in golden.iterrows():

        print(
            f"Judging {index + 1}/{len(golden)}"
        )

        scores = judge.judge(
            customer_message=
                row["customer_text"],
            response=
                row["generated_response"],
            historical_evidence=
                retrieved_evidence[index]
        )

        scores["tweet_id"] = row[
            "tweet_id"
        ] if "tweet_id" in row else index

        judge_rows.append(
            scores
        )

    judge_df = pd.DataFrame(
        judge_rows
    )

    judge_df.to_csv(
        HUMAN_SCORES_PATH,
        index=False
    )

    print()
    print("=" * 60)
    print("LLM JUDGE RESULTS")
    print("=" * 60)

    score_columns = [
        "relevance",
        "correctness",
        "groundedness",
        "helpfulness",
        "tone",
        "overall"
    ]

    for column in score_columns:

        if column in judge_df:

            print(
                f"{column.capitalize():15}: "
                f"{judge_df[column].mean():.2f}/5"
            )

    print()
    print(
        f"Judge scores saved to: "
        f"{HUMAN_SCORES_PATH}"
    )


if __name__ == "__main__":
    evaluate()