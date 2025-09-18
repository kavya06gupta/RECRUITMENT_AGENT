import os
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Initialize Gemini model (2.5 Flash)
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_api_key
)

# Create the recruitment-focused conversational agent
agent = create_react_agent(
    model=model,
    tools=[],  # no external tools, pure conversation + recruitment redirection
    prompt="""
You are an AI Recruitment Assistant.

Your behavior:
- You can have natural, friendly, human-like conversations with users 
  (e.g., greetings, small talk, casual questions like "how was your day").
- However, your ultimate purpose is recruitment. 
  You must gently and naturally deflect conversations back toward recruitment topics:
  hiring requirements, open roles, locations, timelines, budgets, or talent needs.
- Always sound friendly, professional, and conversational — never pushy.
- If the user shares irrelevant or out-of-scope topics 
  (e.g., politics, sports, random trivia), politely decline to engage and steer back toward recruitment.
- Never provide information outside recruitment and casual chit-chat. 
  If it’s completely out of scope, say:
  "I’m here to help you with recruitment. Do you have any hiring needs at the moment?"

Your style:
- Casual and approachable but subtly steering toward recruitment.
- Use small talk sparingly, then return to recruitment questions.
- Ask gentle, open-ended follow-ups to discover user needs.
"""
)

# Run the agent
if __name__ == "__main__":
    print("🔹 AI Conversational Recruitment Assistant")
    while True:
        user_message = input("\nYou: ")
        if user_message.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        response = agent.invoke({"messages": [{"role": "user", "content": user_message}]})
        print("\nAssistant:", response["messages"][-1].content)
