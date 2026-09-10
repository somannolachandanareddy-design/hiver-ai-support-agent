from openai import OpenAI

from .config import (
    OPENAI_API_KEY,
    OPENAI_MODEL
)


SYSTEM_PROMPT = """
You are a customer support response assistant.

Write a concise, professional response to the customer.

Use the historical support examples as evidence for
how similar issues were handled.

Do not invent policies, refunds, timelines, or actions
that are not supported by the evidence.

If the evidence is insufficient, recommend human review
instead of making an unsupported claim.
"""


class ResponseGenerator:

    def __init__(self):

        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not configured."
            )

        self.client = OpenAI(
            api_key=OPENAI_API_KEY
        )

    def generate(
        self,
        customer_message,
        intent,
        retrieved_examples
    ):
        """
        Generate a grounded customer-support response.
        """

        evidence = "\n\n".join(
            [
                f"Example {i + 1}: {item['text']}"
                for i, item in enumerate(
                    retrieved_examples
                )
            ]
        )

        prompt = f"""
Customer message:
{customer_message}

Predicted intent:
{intent}

Historical support evidence:
{evidence}

Draft a helpful, concise and grounded
customer-support response.
"""

        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content.strip()