from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class MajorityClassBaseline:

    def __init__(self):

        self.majority_class = None

    def fit(self, labels):

        counter = Counter(labels)

        self.majority_class = counter.most_common(
            1
        )[0][0]

        return self

    def predict(self, texts):

        if self.majority_class is None:

            raise RuntimeError(
                "Baseline has not been fitted."
            )

        return [
            self.majority_class
            for _ in texts
        ]


class TfidfBaseline:

    def __init__(self):

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=20_000
        )

        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )

    def fit(self, texts, labels):

        features = self.vectorizer.fit_transform(
            texts
        )

        self.model.fit(
            features,
            labels
        )

        return self

    def predict(self, texts):

        features = self.vectorizer.transform(
            texts
        )

        return self.model.predict(
            features
        )