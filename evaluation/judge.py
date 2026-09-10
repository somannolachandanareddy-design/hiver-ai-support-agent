import json

from openai import OpenAI

from src.config import (
    OPENAI_API_KEY,
    OPENAI_MODEL
)


JUDGE_PROMPT = """
You are evaluating an AI customer-support response.

Score the response from 1 to 5 on each dimension.

Dimensions:

1. Relevance
Does the response directly address the customer's issue?

2. Correctness
Is the response factually appropriate given the historical evidence?

3. Groundedness
Does the response avoid unsupported claims and remain grounded
in the supplied historical support examples?

4. Helpfulness
Would this response reasonably help the customer move toward
a resolution?

5. Tone
Is the response professional, concise and appropriate for
customer support?

Return ONLY valid JSON in this format:

{
  "relevance": 1,
  "correctness": 1,
  "groundedness": 1,
  "helpfulness": 1,
  "tone": 1,
  "overall": 1,
  "reason": "short explanation"
}
"""


class LLMJudge:

    def __init__(self):

        if not OPENAI_API_KEY:

            raise ValueError(
                "OPENAI_API_KEY is not configured."
            )

        self.client = OpenAI(
            api_key=OPENAI_API_KEY
        )

    def judge(
        self,
        customer_message,
        response,
        historical_evidence
    ):

        evidence = "\n\n".join(
            [
                f"Example {i + 1}: {item}"
                for i, item in enumerate(
                    historical_evidence
                )
            ]
        )

        prompt = f"""
Customer message:
{customer_message}

AI response:
{response}

Historical support evidence:
{evidence}

Evaluate the AI response using the rubric.
"""

        result = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            response_format={
                "type": "json_object"
            },
            messages=[
                {
                    "role": "system",
                    "content": JUDGE_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        content = (
            result
            .choices[0]
            .message
            .content
        )

        return json.loads(content)