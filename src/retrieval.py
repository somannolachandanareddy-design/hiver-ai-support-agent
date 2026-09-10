import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


class SemanticRetriever:
    """
    Semantic retrieval using Sentence Transformers and FAISS.
    """

    def __init__(
        self,
        model_name="all-MiniLM-L6-v2"
    ):

        self.model = SentenceTransformer(
            model_name
        )

        self.index = None

        self.documents = []

    def build(self, documents):
        """
        Create the FAISS vector index.
        """

        self.documents = list(documents)

        embeddings = self.model.encode(
            self.documents,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(embeddings)

        return self

    def search(self, query, top_k=5):
        """
        Retrieve the most similar historical examples.
        """

        if self.index is None:
            raise RuntimeError(
                "Retriever has not been built."
            )

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index == -1:
                continue

            results.append(
                {
                    "text": self.documents[index],
                    "score": float(score)
                }
            )

        return results