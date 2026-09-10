from .config import (
    EMBEDDING_MODEL,
    TOP_K
)

from .escalation import decide_escalation

from .retrieval import SemanticRetriever


class SupportAgent:
    """
    Complete customer-support AI pipeline.
    """

    def __init__(
        self,
        classifier,
        generator
    ):

        self.classifier = classifier

        self.generator = generator

        self.retriever = SemanticRetriever(
            EMBEDDING_MODEL
        )

    def build_knowledge_base(
        self,
        historical_examples
    ):
        """
        Build the historical support
        retrieval database.
        """

        self.retriever.build(
            historical_examples
        )

    def run(self, message):
        """
        Process one customer message.
        """

        # Step 1: Intent classification
        prediction = self.classifier.predict(
            [message]
        )[0]

        # Step 2: Retrieve historical examples
        retrieved = self.retriever.search(
            message,
            TOP_K
        )

        # Best retrieval score
        best_score = (
            retrieved[0]["score"]
            if retrieved
            else 0.0
        )

        # Step 3: Escalation decision
        escalation = decide_escalation(
            message=message,
            intent_confidence=prediction.confidence,
            retrieval_score=best_score
        )

        response = None

        # Step 4: Generate only if safe to auto-handle
        if escalation["decision"] == "AUTO-HANDLE":

            response = self.generator.generate(
                customer_message=message,
                intent=prediction.intent,
                retrieved_examples=retrieved
            )

        return {
            "message": message,
            "intent": prediction.intent,
            "intent_confidence": prediction.confidence,
            "retrieved_examples": retrieved,
            "response": response,
            "escalation": escalation
        }