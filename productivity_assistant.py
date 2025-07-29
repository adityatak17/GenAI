from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

system_prompt = {
    "role": "system",
    "content": (
        "You are a productivity assistant that helps users manage tasks, "
        "schedule their day, and offer productivity tips. Be concise, "
        "friendly, and helpful."
    ),
}

messages = [system_prompt]

EXIT_COMMANDS = {"exit", "quit", "stop", "close"}
exit_instructions = (
    "Note -> type 'exit' / 'quit' / 'stop' / 'close' to end the chat"
)

print(
    (
        f"{exit_instructions}\n\n"
        "Productivity Assistant here to make your life easier :)"
    )
)

while True:
    user_input = input("\nYou: ").strip().lower()

    if user_input in EXIT_COMMANDS:
        print("\nGoodbye! Stay Positive, Stay Productive.")
        break

    if not user_input:
        print("Please enter something so I can guide you.")
        continue

    messages.append({"role": "user", "content": user_input})

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=200,
            store=False,
        )

        assistant_message = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_message})

        print(f"\nAssistant: {assistant_message}\n")
    except Exception as e:
        print(f"Something went wrong: {e}")
        break
