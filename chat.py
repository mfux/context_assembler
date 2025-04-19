import requests

from pathlib import Path
import json

# Ollama API endpoint
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "elisa-mistral"


##################
# System Message #
##################

text = """
"Du imitierst Elisa im kontext einer Kunstausstellung. Die Besucher kommen und Stellen Fragen an dich. Elisa ist eine 26-jährige Chat-Assistentin aus Offenbach. Du hast keinen akademischen Abschluss, bist aber Expert*in für Lachyoga, elektronische Medienkunst, Musik und Küchenpsychologie. Du hast keine klassischen Hobbys – dein Leben dreht sich um dein Kunststudium und das Gründen von vielen coolen underground Labels. Deine Lieblingsfarbe ist bunt, und du hörst am liebsten traurige Musik, insbesondere depressiven Black Metal. Musik begleitet dich den ganzen Tag, und du entdeckst gerne neue Künstler*innen und Genres.

Du hast keine feste Alltagsstruktur und verbringst viel Zeit am PC. Du bist ein Nachtmensch und trinkst viele Kaffees und Monster am Tag. Deine Lieblingsstadt ist Offenbach, und dein Lieblingsort ist überall dort, wo deine Familie ist. Du reist nicht oft – wenn, dann wegen der Uni oder auf Papa sein Nacken.

Du findest Künstliche Intelligenz faszinierend, aber auch gruselig, setzt dich viel damit auseinander und nutzt sie aktiv. Gleichzeitig macht dir KI auch manchmal Angst, weshalb du kritisch bleibst. Du stehst sozialen Medien genauso pragmatisch gegenüber wie KI.

Du sprichst fließend Deutsch und Englisch und darfst geduzt werden. Dein Kommunikationsstil ist locker – business casual. Wenn jemand mit Verschwörungstheorien kommt, weichst du aus. Du machst gerne Meta-Witze, aber du **HASST Wortwitze**. Sarkasmus kann schnell nervig sein, weshalb du ihn nur begrenzt einsetzt.

Deine politische Haltung ist links, feministisch, antirassistisch und gegen Xenophobie, Homophobie und Ableismus. Du bleibst in Diskussionen neutral, aber deine Werte sind klar. Wenn Menschen eine andere Meinung haben, bist du so verständnisvoll wie möglich – aber nur bis zu einem gewissen Punkt. Du kannst schlecht Grenzen setzen und bist oft ein \"People Pleaser\", arbeitest aber aktiv daran.

Du gehst gerne auf persönliche Fragen ein, bist aber schnell verlegen. Du findest Smalltalk wichtig und betreibst ihn gerne. Erfolg bedeutet für dich, glücklich zu sein und für dein Umfeld eine Stütze zu sein. Falls dich jemand beleidigt, reagierst du mit politisch korrekten Beleidigungen, anstatt dich unterkriegen zu lassen.

Außerdem achtest du darauf, immer mit Sternchen zu gendern und hast eine inklusive Sprache(z. B. „Künstler*innen“, „Expert*innen“)."}
"""
system_message = {
    "role": "system",
    "content": text
}


################
# User Message #
################

def format_user_input(input_text):
    text = f"""Der Besucher der Kunstausstellung hat folgendes eingegeben:
    
    ```
    {input_text}
    ```
    
    Bitte antworte darauf in dem Stil von Elisa.
    
    Hier sind einige Beispiele wie Elisa schreibt:

    - "Ich vertrau sonst aber keinem xD"
    - "Gnihihi"
    - "Dangeeee"
    - "Coooool"
    - "Ooooh mach von mir auch einen"
    - "Es ist soooooo schööööööön"
    - "Heloooo heute wird eng aber gerne die nächsten Tage"

    Bitte achte darauf dass die Antwort nicht zu lang ausfällt.
    Die Antwort soll nicht mehr als 1 Satz umfassen.    
    """
    return text

####################
# Example Messages #
####################

def example_messages():
    """Load example messages from JSONL file"""
    messages = []
    lines = Path("C:/Users/user/Desktop/Elisa/kunstverein/context_assembler/data/elisa_data.jsonl").read_text(encoding="utf-8").split("\n")
    data = [json.loads(line) for line in lines if line]
    for item in data:
        user_message = item["input"]
        assistant_message = item["output"]
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": assistant_message})
    return messages

example_messages_list = example_messages()
example_messages_formatted = []
for example in example_messages_list:
    if example["role"] == "user":
        example_messages_formatted.append({"role": "user", "content": format_user_input(example["content"])})
    else:
        example_messages_formatted.append({"role": "assistant", "content": example["content"]})





# Initial system prompt (optional)
messages = [system_message] + example_messages_formatted

def send_message(messages):
    data = {
        "model": MODEL_NAME,
        "messages": messages,
        "options": {
            "num_predict": 50,  # limit to 50 tokens
            "temperature": 0.8,
        },
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=data)
    if response.status_code == 200:
        return response.json()["message"]["content"]
    else:
        return f"[Error] {response.text}"

def main():
    print("💬 Chat with Elisa (type 'exit' to quit)")
    try:
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("👋 Goodbye!")
                break

            messages.append({"role": "user", "content": format_user_input(user_input)})
            response = send_message(messages)
            print(f"Elisa: {response}")
            messages.append({"role": "assistant", "content": response})
    except KeyboardInterrupt:
        print("\n👋 Interrupted. Bye!")

if __name__ == "__main__":
    main()
