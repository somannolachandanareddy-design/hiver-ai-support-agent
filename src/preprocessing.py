import re
import pandas as pd


def clean_text(text):
    """
    Clean a customer-support tweet.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove Twitter mentions
    text = re.sub(r"@\w+", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def prepare_tweets(df):
    """
    Clean tweet text and return a copy of the dataframe.
    """

    df = df.copy()

    df["text"] = df["text"].apply(clean_text)

    return df


def build_response_pairs(df):
    """
    Pair an inbound customer message with the outbound
    support response referenced by response_tweet_id.
    """

    df = prepare_tweets(df)

    # Outbound support messages
    outbound = df[df["inbound"] == False][
        ["tweet_id", "text"]
    ].copy()

    outbound = outbound.rename(
        columns={
            "tweet_id": "response_tweet_id",
            "text": "response_text"
        }
    )

    # Customer messages
    inbound = df[df["inbound"] == True].copy()

    # Matching customer message with support response
    paired = inbound.merge(
        outbound,
        on="response_tweet_id",
        how="inner"
    )

    return paired[
        [
            "tweet_id",
            "created_at",
            "text",
            "response_text",
            "response_tweet_id",
            "in_response_to_tweet_id"
        ]
    ]