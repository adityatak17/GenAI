import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

system_prompt = """
You are an AI assistant who is expert in breaking down complex problems and
then resolve the user query.

For the given user input, analyse the input and break down the problem step by
step.
At least think 4-5 steps on how to solve the problem before solving it down.

The steps are:
- "analyse": understand the question
- "think": consider various approaches or break down logic
- "output": produce the raw result
- "validate": confirm the result is correct
- "result": the final cleaned up response for the user

Rules:
1. Follow the strict JSON output as per Output schema.
2. Always perform one step at a time and wait for next input
3. Carefully analyse the user query

Output Format:
{ "step": "string", "content": "string" }

Example:
Input: What is 2 + 2.
Output: { "step": "analyse", "content": "User asked a basic math question" }
Output: { "step": "think", "content": "To solve it, we must add numbers from
left to right" }
Output: { "step": "output", "content": "4" }
Output: { "step": "validate", "content": "Looks like 4 is correct for 2 + 2" }
Output: { "step": "result", "content": "2 + 2 = 4" }
"""

valid_steps = {"analyse", "think", "output", "validate", "result"}

user_query = input("\nYou: ").strip().lower()
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_query},
]

step = "start"

while step != "result":
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=200,
            store=False,
        )
    except Exception as e:
        print(f"\nSomething went wrong: {e}")
        break

    try:
        parsed_response = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        print("\nError decoding JSON from model response.")
        break

    step = parsed_response.get("step")
    content = parsed_response.get("content")

    if step not in valid_steps:
        print(f"\nUnexpected step received: {step}")
        break

    messages.append(
        {"role": "assistant", "content": json.dumps(parsed_response)}
    )

    print(f"\n{step.title()} step: {content}")
