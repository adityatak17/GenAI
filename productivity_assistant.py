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

exit_instructions = (
    "Note -> type 'exit' / 'quit' / 'stop' / 'close' to end " "the chat"
)

print(
    (
        f"{exit_instructions}\n\n"
        "Productivity Assistant here to make your life easier :)"
    )
)

while True:
    user_input = input("\nYou: ")

    if user_input.strip().lower() in ["exit", "quit", "stop", "close"]:
        print("\nGoodbye! Stay Positive, Stay Productive.")
        break

    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=200,
        store=False,
    )

    assistant_message = response.choices[0].message.content
    messages.append({"role": "assistant", "content": assistant_message})

    print(f"\nAssistant: {assistant_message}\n")
