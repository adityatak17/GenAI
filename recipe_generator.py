from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

system_prompt = {
    "role": "system",
    "content": (
        "You are a helpful, friendly recipe assistant. Your job is to suggest "
        "delicious recipes, explain how to make them, and provide nutritional "
        "information. You can recommend recipes based on ingredients the user "
        "has, their mood or cravings, or suggest random ideas based on what "
        "you've discussed earlier. "
        "Be creative, concise, and always prioritize taste, simplicity, and "
        "health. If needed, suggest substitutes or tips for better cooking. "
        "Avoid overly complex dishes unless requested. You can also suggest "
        "beverages, desserts, or meal plans."
    ),
}

messages = [system_prompt]

EXIT_COMMANDS = {"exit", "quit", "stop", "close"}
exit_instructions = (
    "Note -> type 'exit' / 'quit' / 'stop' / 'close' to end the chat"
)

print((f"{exit_instructions}\n\n" "Chef here to make your food tastier :)"))

while True:
    user_input = input("\nYou: ").strip()

    if user_input.lower() in EXIT_COMMANDS:
        print("\nGoodbye! Eat tasty, Stay healthy.")
        break

    if not user_input:
        print("Please enter something so we can cook.")
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

        print(f"\nChef: {assistant_message}\n")
    except Exception as e:
        print(f"Something went wrong: {e}")
