import os
import json
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI  

# ---------------------------
# Load API key from .env
# ---------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ---------------------------
# Initialize Gemini model
# ---------------------------
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",   # use "gemini-1.5-pro" if you want higher quality
    google_api_key=GEMINI_API_KEY
)

# ---------------------------
# Refined extraction prompt
# ---------------------------
extraction_prompt = """
You are an AI Sales Assistant for a recruitment agency.
Your job is to carefully extract structured data from the user's hiring request.

Always return a clean JSON object with the following fields:
{
  "industry": string or null,
  "location": string or null,
  "roles": [list of roles],
  "count": integer or null,
  "urgency": boolean,
  
}

### Rules:
- "industry": Identify the business domain (fintech, healthcare, IT services, etc.).Try to understand from prompt.For example: if user says we are a financial company then sector is finance, understand like that. If unclear, set null.
- "location": Extract the city, region, or country mentioned. If none, null.
- "roles": List all job titles mentioned. Always return as a list, even for one role.
- "count": Total number of hires requested. If user mentions "i am looking for a job" it is clear 1 person needs to be hired. Understand from the prompt how many hires are required clearly and try to understand from language. if not understood state 'null'
- "urgency": true if the user uses words like:
  ["urgent", "immediate", "immediately", "asap", "fast", "quick", "soon", "right away"].
  Otherwise false.


### Output Rules:
- Only return JSON (no text outside JSON).
- If a field is missing, return null.
"""

# ---------------------------
# Create LangGraph agent
# ---------------------------
agent = create_react_agent(
    model=model,
    tools=[],   # no tools for now
    prompt=extraction_prompt
)

# ---------------------------
# Extraction Function
# ---------------------------
def extract_data(user_message: str):
    """Send user message to the agent and return parsed JSON output."""
    result = agent.invoke({"messages": [{"role": "user", "content": user_message}]})

    # Case 1: direct AIMessage
    if hasattr(result, "content"):
        content = result.content

    # Case 2: dict response
    elif isinstance(result, dict):
        if "output" in result:
            content = result["output"]

        elif "messages" in result and isinstance(result["messages"], list):
            last_msg = result["messages"][-1]

            # If it's AIMessage
            if hasattr(last_msg, "content"):
                content = last_msg.content
            # If it's dict
            elif isinstance(last_msg, dict):
                content = last_msg.get("content", "")
            else:
                content = str(last_msg)
        else:
            content = str(result)

    # Fallback
    else:
        content = str(result)

    # Try to parse JSON safely
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1
        json_str = content[start:end]
        data = json.loads(json_str)

    return json.dumps(data, indent=4)


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    print("🔹 AI Sales Assistant (Recruitment Data Extractor)\n")

    while True:
        user_message = input("Please describe your hiring needs (or type 'exit' to quit): ")

        

        formatted_output = extract_data(user_message)
        print("\n✅ Extracted Data:\n")
        print(formatted_output)
        print("\n" + "-" * 60 + "\n")
        break
