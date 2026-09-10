from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


@dataclass
class IntentPrediction:
    intent: str
    confidence: float


class IntentClassifier:
    """
    TF-IDF + Logistic Regression intent classifier.
    """

    def __init__(self):

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=20000
        )

        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )

    def fit(self, texts, labels):
        """
        Train the intent classifier.
        """

        features = self.vectorizer.fit_transform(texts)

        self.model.fit(features, labels)

        return self

    def predict(self, texts):
        """
        Predict intent and confidence.
        """

        features = self.vectorizer.transform(texts)

        probabilities = self.model.predict_proba(features)

        predictions = self.model.classes_[
            probabilities.argmax(axis=1)
        ]

        results = []

        for i, label in enumerate(predictions):

            confidence = float(
                probabilities[i].max()
            )

            results.append(
                IntentPrediction(
                    intent=label,
                    confidence=confidence
                )
            )

        return results