from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "twcs.csv"

CHUNK_SIZE = 100_000


def analyze_dataset():

    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        print("Place twcs.csv inside the data/ folder.")
        return

    total_rows = 0
    inbound_count = 0
    outbound_count = 0

    author_counts = Counter()
    author_examples = defaultdict(list)

    print("Analyzing TWCS dataset...")
    print(f"Dataset: {DATA_PATH}")
    print()

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            DATA_PATH,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ),
        start=1
    ):

        total_rows += len(chunk)

        inbound_count += int(
            (chunk["inbound"] == True).sum()
        )

        outbound = chunk[
            chunk["inbound"] == False
        ]

        outbound_count += len(outbound)

        # Count support accounts
        counts = outbound["author_id"].value_counts()

        for author_id, count in counts.items():
            author_counts[author_id] += int(count)

        # Keep a few representative outbound examples
        for _, row in outbound.iterrows():

            author_id = row["author_id"]

            if len(author_examples[author_id]) < 3:

                author_examples[author_id].append(
                    str(row["text"])
                )

        print(
            f"Processed chunk {chunk_number} | "
            f"Rows: {total_rows:,}"
        )

    print()
    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(f"Total rows      : {total_rows:,}")
    print(f"Inbound tweets  : {inbound_count:,}")
    print(f"Outbound tweets : {outbound_count:,}")

    print()
    print("=" * 70)
    print("TOP SUPPORT ACCOUNTS")
    print("=" * 70)

    for rank, (author_id, count) in enumerate(
        author_counts.most_common(20),
        start=1
    ):

        print()
        print(f"{rank}. Author ID: {author_id}")
        print(f"   Outbound tweets: {count:,}")

        examples = author_examples[author_id]

        for example in examples:
            print(f"   Example: {example[:250]}")


if __name__ == "__main__":
    analyze_dataset()