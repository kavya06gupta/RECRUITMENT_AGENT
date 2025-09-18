# main.py
import os
import json
from dotenv import load_dotenv
from data_extraction import extract_data        # <-- your extractor
from normal_conversation import agent as normal_agent
from recommendation import recommend_services   # <-- recruitment recommender

# ---------------------------
# Load API Key
# ---------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ---------------------------
# Classifier Prompt (rule-based for now)
# ---------------------------
def classify_from_data(extracted: dict) -> str:
    """
    Classify based on extracted structured data:
    - If all fields are null/empty → normal conversation
    - If any useful hiring info present → recruitment
    """
    if (
        not extracted.get("industry")
        and not extracted.get("location")
        and not extracted.get("roles")
        and not extracted.get("count")
        and not extracted.get("urgency")
    ):
        return "normal"
    return "recruitment"

# ---------------------------
# Super Agent Flow
# ---------------------------
if __name__ == "__main__":
    print("Hi, I am a recruiting agent, How can I help you?")
    while True:
        user_message = input("\nYou: ")
        if user_message.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        # Step 1: Extract data
        extracted_json = extract_data(user_message)
        extracted = json.loads(extracted_json)

        # Step 2: Classify intent
        intent = classify_from_data(extracted)

        # Step 3: Route to correct agent
        if intent == "normal":
            response = normal_agent.invoke({"messages": [{"role": "user", "content": user_message}]})
            print("\nAssistant:", response["messages"][-1].content)

        elif intent == "recruitment":
            print("\n📌 Extracted Recruitment Data:")
            print(json.dumps(extracted, indent=4))

            recommendations = recommend_services(user_message)
            print("\nAssistant:", recommendations)
