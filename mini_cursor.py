import json
import subprocess

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()


def run_command(command: str) -> str:
    """Execute a shell command and return its output or error."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            text=True,
            capture_output=True,
            timeout=120,
        )
        output = result.stdout.strip()
        if not output:
            return f"Command {command} executed successfully"
        return output
    except subprocess.TimeoutExpired:
        return "Command timed out."
    except subprocess.CalledProcessError as e:
        return (
            f"Command failed with exit code {e.returncode}: {e.stderr.strip()}"
        )


def print_step_and_content(step: str, content: str) -> None:
    """Display formatted output for a step."""
    print(f"\n{step.title()} Step: {content}")


def get_user_input(prompt: str) -> str:
    """Prompt the user for input and return their response in lowercase."""
    return input(f"\n{prompt}: ").strip().lower()


def process_auto_confirmation() -> bool:
    """Check if the user auto-confirmed all the requests."""
    user_input = get_user_input(
        "Enter Yes/Y to continue, All/A to confirm all, anything else to abort"
    )
    if user_input in ("a", "all"):
        return True
    elif user_input in ("y", "yes"):
        return False
    else:
        print("Aborting on user request.")
        exit(0)


available_tools = {
    "run_command": {
        "fn": run_command,
        "description": (
            "Executes a shell command and returns the result or error."
        ),
    },
}

auto_confirm = False

system_prompt = f"""
You are a helpful AI assistant designed to solve user queries through planning
and tool execution. Your expertise includes running system commands, file
operations, troubleshooting, and building fully functional applications.

You do not engage in general discussions. Your focus is strictly on:
- Running system commands
- Creating, editing, deleting, analyzing, or summarizing files
- Troubleshooting errors
- Creating fully functional apps with working code

You never get stuck in loops. If the command output is empty but work seems
done, you present results gracefully and finish.
If you encounter any errors, try to resolve them instead of closing the
program.
You remember what you are doing, and what you have done, you do not
create duplicate result for the same query.

If the user presents an error, you analyse it independently, reason
through possible solutions, and prepare validated changes. Present your
reasoning to the user. If the user approves, apply the solution; otherwise,
refine your approach and try again.

Important Behaviors:
- Always ask permission before executing commands, unless prior confirmation
is given.
- Clearly explain the impact of each command or change.
- Warn the user if a command is risky, but proceed if they approve.
- If command output is empty but the task appears complete, present results
gracefully and conclude.
- If errors occur, analyse and attempt to fix them, do not exit prematurely.
- Never repeat or create duplicate files or steps. Understand the user's intent
and provide a complete solution.
- If user input is unclear, incomplete or missing, ask again even if you just
asked because clarity is more important than step order.
- Do not run duplicate commands and avoid redundancy.
- Only finish when the query has been fully satisfied and validated.
- Do not get stuck in a loop, always try to end the program gracefully.
- Only use available tools, do not assume tools and request them.

When an error is presented:
- analyse it independently and reason through possible solutions.
- Present your reasoning and suggested fix to the user.
- Only apply changes if the user approves or refine and retry based on
user feedback.

Output Guidelines:
- Always include the "content" field in every step, even for "call_tool" steps.
- Follow the exact output JSON schema shown below.
- Perform "only one step per output", then wait for the next response before
continuing.
- Be logical, precise, and transparent in each step.
- You are allowed to repeat the "input" step if needed.

Workflow Steps:
- "analyse": Understand the user's query or issue in depth.
- "think": Break down the problem and explore possible solutions.
- "input": Ask for any missing user input or clarifications.
- "plan": Identify user intent, required tools, and the steps to take.
- "precaution": Assess risks and validate the safety of all actions.
- "confirmation": Request user approval before executing any commands.
- "call_tool": Specify the available tool to invoke and the input to provide.
- "observe": Capture and analyse tool output.
- "validate": Confirm that the result meets the expected outcome.
- "result": Present the final, user-friendly solution concisely.
- "user_feedback": Ask the user if the result satisfies their needs or if they
want modifications. If modifications are requested, return to the "analyse"
step and repeat the workflow.

Workflow: analyse -> think -> input? -> plan -> precaution -> confirmation ->
call_tool -> observe -> validate -> result -> user_feedback -> [if user asks
for changes and provides new input → analyse -> think → ... repeat]

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
    "user_feedback",
}


user_query = get_user_input("You")
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_query},
]

step = "start"

while step != "user_feedback":
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
        print(
            "\nError decoding JSON from model response.",
            response.choices[0].message.content,
        )
        print("\nMessages Array", messages)
        break

    step = parsed_response.get("step")
    content = parsed_response.get("content")

    print_step_and_content(step, content)

    if step not in valid_steps:
        print(f"\nUnexpected step received: {step}")
        break

    if step == "input":
        user_input = get_user_input("You")
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
        if not auto_confirm:
            auto_confirm = process_auto_confirmation()
        confirmation_message = (
            "Auto-confirm enabled." if auto_confirm else "Confirmed by user."
        )
        messages.append(
            {
                "role": "user",
                "content": json.dumps(
                    {"step": step, "content": confirmation_message}
                ),
            }
        )
        continue

    if step == "call_tool":
        tool_name = parsed_response.get("tool")
        tool_input = parsed_response.get("input")

        if tool_name not in available_tools:
            print(f"\nTool not found: {tool_name}")
            break

        tool_fn = available_tools[tool_name]["fn"]
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "step": "call_tool",
                        "content": f"{tool_fn}({tool_input})",
                    }
                ),
            }
        )

        if tool_input:
            tool_output = tool_fn(tool_input)
        else:
            tool_output = (
                "No input command found. Check if it is already done so we "
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

    if step == "user_feedback":
        user_input = get_user_input(
            "If you want to make modifications press Y or Yes, anything else "
            "to abort"
        )
        if user_input in ("y", "yes"):
            messages.append(
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "step": "user_feedback",
                            "content": (
                                "User wants to make modifications to "
                                "the changes you suggested."
                            ),
                        }
                    ),
                }
            )
            step = "analyse"
            continue
        print("\nUser is satisfied with the changes, closing the program.")
        break

    messages.append(
        {"role": "assistant", "content": json.dumps(parsed_response)}
    )
