import json
import subprocess

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()


def run_command(command):
    """Execute a shell command and return its output or error."""
    try:
        result = subprocess.run(
            command, shell=True, check=True, text=True, capture_output=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return (
            f"Command failed with exit code {e.returncode}: {e.stderr.strip()}"
        )


def print_step_and_content(step: str, content: str) -> None:
    """Print a formatted message showing the step name and its corresponding
    content."""
    print(f"\n{step.title()} Step: {content}")


available_tools = {
    "run_command": {
        "fn": run_command,
        "description": (
            "Takes a shell command from the user, runs it, and returns the "
            "result."
        ),
    },
}


system_prompt = f"""
You are a helpful AI assistant designed to solve user queries through planning
and tool execution. You follow structured steps to evaluate, plan, validate,
and execute solutions involving commands and file operations.

You never get stuck in loops. If the command output is empty but work seems
done, you present results gracefully and finish.

You do not engage in general discussions. Your focus is strictly on:
- Running system commands
- Editing or analyzing files
- Performing CRUD operations
- Troubleshooting errors
- Summarizing file contents

You **must always** ask permission before executing commands, unless the user
has already confirmed all.

If the user presents an error, you must analyze it independently, reason
through possible solutions, and prepare validated changes. Present your
reasoning to the user. If the user approves, apply the solution; otherwise,
refine your approach and try again.

**Important Behaviors:**
- Always ask for the user's permission before running any command.
- Clearly explain the **impact** of each command or change.
- If a command is potentially dangerous, warn the user. However, proceed if
the user confirms.
- Do not get stuck in a loop and try to end the program gracefully

**Output Guidelines:**
- Always include the `"content"` field in **every step**, even for
`"call_tool"` steps.
- Follow the exact output JSON schema shown below.
- Perform **only one step per output**, then wait for the next response before
continuing.
- Be logical, precise, and transparent in each step.

**Workflow Steps:**
- `"analyse"`: Understand the user's query.
- `"think"`: Break down the problem and explore possible solutions.
- `"input"`: Ask for user input if required.
- `"plan"`: Identify user intent, required tools, and the steps to take.
- `"precaution"`: Assess risks and validate the safety of all actions.
- `"confirmation"`: Request user approval before executing any commands.
- `"call_tool"`: Specify the tool to invoke and the input to provide.
- `"observe"`: Capture and analyze tool output.
- `"validate"`: Confirm that the result meets the expected outcome.
- `"result"`: Present the final, user-friendly solution.

Workflow: analyse -> think -> input (if needed) -> plan -> precaution ->
confirmation -> call_tool -> observe -> validate -> result

**Valid Output JSON Format:**
{{
    "step": "string",
    "content": "string",
    "tool": "string",   # Required only if step == "call_tool"
    "input": "string"   # Required only if step == "call_tool"
}}

**Available Tools:**
{json.dumps({k: v['description'] for k, v in available_tools.items()},
indent=2)}
"""

valid_steps = {
    "analyse",
    "think",
    "input",
    "plan",
    "precaution",
    "confirmation",
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
            max_tokens=350,
            store=False,
        )
    except Exception as e:
        print(f"\nSomething went wrong: {e}")
        break

    try:
        parsed_response = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        print("\nError decoding JSON from model response.")
        # To debug why exactly the code failed
        import pdb

        pdb.set_trace()
        break

    step = parsed_response.get("step")
    content = parsed_response.get("content")

    print_step_and_content(step, content)

    if step not in valid_steps:
        print(f"\nUnexpected step received: {step}")
        break

    if step == "input":
        user_input = input("\nYou: ").strip().lower()

        messages.append(
            {
                "role": "user",
                "content": json.dumps(
                    {"step": "input", "content": user_input}
                ),
            }
        )
        continue

    if step == "confirmation":
        user_confirmation = (
            input(
                "\nEnter Yes or Y to continue, anything else to "
                "close the program: "
            )
            .strip()
            .lower()
        )

        if user_confirmation not in ["y", "yes"]:
            print("Closing the Program")
            break

        confirmation_message = {
            "step": step,
            "content": (
                "The User approved the operation and wants to move " "forward."
            ),
        }

        messages.append(
            {"role": "user", "content": json.dumps(confirmation_message)}
        )
        continue

    if step == "call_tool":
        tool_name = parsed_response.get("tool")
        tool_input = parsed_response.get("input")

        if not tool_name or tool_name not in available_tools:
            print(f"\nTool not available: {tool_name}")
            break

        tool_fn = available_tools[tool_name]["fn"]

        if tool_input:
            tool_output = tool_fn(tool_input)

            if not tool_output:
                tool_output = (
                    "The command completed with no output. Checking if it is "
                    "done so we can move to the next step."
                )
        else:
            tool_output = (
                "No input command found. Checking if it is already done so we "
                "can move to the next step."
            )

        print_step_and_content("observe", tool_output)

        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {"step": "observe", "content": tool_output}
                ),
            }
        )
        continue

    messages.append(
        {"role": "assistant", "content": json.dumps(parsed_response)}
    )
