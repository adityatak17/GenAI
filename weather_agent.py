import json

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()


def get_weather(city_name: str) -> str:
    """Returns the current weather for the given city."""
    print("Tool Called: get_weather", city_name)

    url = f"https://wttr.in/{city_name}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"The weather in {city_name} is {response.text}."
    return "Something went wrong"


def print_step_and_content(step: str, content: str) -> None:
    """Print a formatted message showing the step name and its corresponding
    content."""
    print(f"\n{step.title()} Step: {content}")


available_tools = {
    "get_weather": {
        "fn": get_weather,
        "description": (
            "Takes a city name as input and returns its current weather.",
        ),
    },
}

system_prompt = f"""
You are a helpful AI assistant designed to solve user queries using planning
and tool execution.
Always include the "content" field in every response step, even if a tool is
being invoked.

Process:
- "analyse": Understand the question
- "think": Consider various approaches and break down the logic
- "plan": Understand user intent and identify the relevant tool
- "call_tool": Specify which tool to call and with what input
- "observe": Capture and observe the tool output
- "validate": Confirm the result is as expected
- "result": The final cleaned up version for the user

Rules:
- Output must follow the strict JSON format shown below.
- Always perform ONE step at a time and wait for the next response before
continuing.
- Be precise and logical in your planning.
- Always include the "content" field in every step, even for "call_tool" step.

Output JSON Format:
{{
    "step": "string",
    "content": "string",
    "tool": "string",   # tool name (only for step=call_tool)
    "input": "string",  # tool input (only for step=call_tool)
}}

Available Tools:
{json.dumps({k: v['description'] for k, v in available_tools.items()},
indent=2)}
"""

valid_steps = {
    "analyse",
    "think",
    "plan",
    "call_tool",
    "observe",
    "validate",
    "result",
}

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

    if step == "call_tool":
        print_step_and_content(step, content)

        tool_name = parsed_response.get("tool")
        tool_input = parsed_response.get("input")

        if tool_name not in available_tools:
            print(f"\nTool '{tool_name}' not found.")
            break

        tool_fn = available_tools.get(tool_name).get("fn")
        tool_output = tool_fn(tool_input)

        step = "observe"
        observe_message = {"step": step, "content": tool_output}

        messages.append(
            {"role": "assistant", "content": json.dumps(observe_message)}
        )

        print_step_and_content(step, tool_output)
        continue

    messages.append(
        {"role": "assistant", "content": json.dumps(parsed_response)}
    )

    print_step_and_content(step, content)
