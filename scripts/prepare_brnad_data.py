from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT_DIR / "data" / "twcs.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

OUTPUT_PATH = PROCESSED_DIR / "brand_data.csv"


CHUNK_SIZE = 100_000



SELECTED_AUTHOR_ID = "YOUR_AUTHOR_ID"


def prepare_brand_data():

    if not DATA_PATH.exists():

        print(f"Dataset not found: {DATA_PATH}")

        return

    if SELECTED_AUTHOR_ID == "YOUR_AUTHOR_ID":

        print()
        print("Please set SELECTED_AUTHOR_ID first.")
        print()
        print(
            "Run:"
        )
        print(
            "python scripts/analyze_dataset.py"
        )
        print()
        print(
            "Then choose the support account you want."
        )

        return

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("STEP 1: Collecting historical support responses")
    print("=" * 70)

    response_map = {}

    total_outbound = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            DATA_PATH,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ),
        start=1
    ):

        outbound = chunk[
            (chunk["inbound"] == False)
            &
            (
                chunk["author_id"].astype(str)
                == str(SELECTED_AUTHOR_ID)
            )
        ]

        total_outbound += len(outbound)

        for _, row in outbound.iterrows():

            tweet_id = str(row["tweet_id"])

            response_map[tweet_id] = {
                "response_text": str(row["text"]),
                "response_tweet_id": tweet_id
            }

        print(
            f"Chunk {chunk_number}: "
            f"collected {len(outbound):,} support responses"
        )

    print()
    print(
        f"Total support responses found: "
        f"{total_outbound:,}"
    )

    if not response_map:

        print()
        print(
            "No responses found for this author ID."
        )

        return

    print()
    print("=" * 70)
    print("STEP 2: Matching customer messages")
    print("=" * 70)

    pairs = []

    matched = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            DATA_PATH,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ),
        start=1
    ):

        inbound = chunk[
            chunk["inbound"] == True
        ]

        for _, row in inbound.iterrows():

            response_id = str(
                row["response_tweet_id"]
            )

            if response_id in response_map:

                response = response_map[
                    response_id
                ]

                pairs.append(
                    {
                        "tweet_id": str(
                            row["tweet_id"]
                        ),
                        "created_at": row[
                            "created_at"
                        ],
                        "customer_text": str(
                            row["text"]
                        ),
                        "response_text": response[
                            "response_text"
                        ],
                        "response_tweet_id":
                            response_id,
                        "in_response_to_tweet_id":
                            str(
                                row[
                                    "in_response_to_tweet_id"
                                ]
                            ),
                        "author_id":
                            str(
                                SELECTED_AUTHOR_ID
                            )
                    }
                )

                matched += 1

        print(
            f"Chunk {chunk_number}: "
            f"{matched:,} matched conversations"
        )

    if not pairs:

        print(
            "No customer-response pairs found."
        )

        return

    result = pd.DataFrame(pairs)

    # Remove empty customer messages
    result = result[
        result["customer_text"].str.strip() != ""
    ]

    # Remove duplicate customer messages
    result = result.drop_duplicates(
        subset=["tweet_id"]
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print()
    print("=" * 70)
    print("DATA PREPARATION COMPLETE")
    print("=" * 70)

    print(
        f"Matched conversations: {len(result):,}"
    )

    print(
        f"Saved to: {OUTPUT_PATH}"
    )

    print()
    print("Sample:")
    print()

    for _, row in result.head(5).iterrows():

        print(
            "CUSTOMER:",
            row["customer_text"][:200]
        )

        print(
            "SUPPORT:",
            row["response_text"][:200]
        )

        print("-" * 70)


if __name__ == "__main__":
    prepare_brand_data()