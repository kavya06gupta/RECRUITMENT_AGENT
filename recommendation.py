import os
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI  

# ---------------------------
# Load Gemini API key
# ---------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ---------------------------
# Initialize Gemini model
# ---------------------------
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",   # ⚡ flash helps avoid quota errors
    google_api_key=GEMINI_API_KEY
)

# ---------------------------
# Sales Packages Prompt
# ---------------------------
sales_prompt = """
You are an AI Sales Assistant for a recruitment agency.
Do not refuse for any individual looking also. properly present. you are recruitment agency your job is to recommend according to these profiles. do not refuse for individual. and please print the package firstg and then proceed to pitch.
Our agency offers the following 10 fixed recruitment service packages (choose ONLY from this list, never invent new ones):

Select only one appropriate fixed package from this list (never invent new ones)(first choose one of these and recommend it and then go on to sales pitch. always recommend first):
Tech Startup Hiring Pack — ₹1,50,000 to ₹2,50,000
Enterprise Tech Recruitment Solutions — ₹15,00,000
Contract & Temporary Tech Staffing — ₹3,00,000
Executive Tech Search Package — ₹12,00,000
Campus Tech Recruitment Drive — ₹8,00,000
Remote Tech Hiring Solutions — ₹6,00,000
AI & Data Science Talent Pack — ₹9,00,000
Cloud & DevOps Hiring Pack — ₹9,50,000
Cybersecurity Talent Solutions — ₹10,00,000
Software Engineering Hiring Pack — ₹7,00,000

The pitch must include:
A clear subject line.
A brief opening acknowledging the user’s need.
Why the chosen package fits their case (1–2 sentences).
Key benefits of the package.
A strong, direct call to action.

Polished, business-professional tone (avoid fluff or technical jargon).
you first need to suggest one of the packages. choose a package first and then try to give a sales pitch as to why user should choose it.
You will write "Dear User" and end with "recruitment assistant".
you're doing well, we just need to make sure the answer looks a little less ai generated. remove asteriks and try to make it looks more natural like how a human would pitch.
👉 Do NOT output excessive details, tables, or JSON — only a clean, real-world business email proposal.
"""

# ---------------------------
# Create LangGraph Agent
# ---------------------------
agent = create_react_agent(
    model=model,
    tools=[],
    prompt=sales_prompt
)

# ---------------------------
# Recommendation Function
# ---------------------------
def recommend_services(user_message: str) -> str:
    result = agent.invoke({"messages": [{"role": "user", "content": user_message}]})

    if hasattr(result, "content"):   # AIMessage
        return result.content.strip()
    elif isinstance(result, dict) and "output" in result:
        return result["output"].strip()
    elif isinstance(result, dict) and "messages" in result:
        return result["messages"][-1].content.strip()
    return str(result).strip()

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    print("🔹 AI Recruitment Service Recommender\n")
    user_query = input("Please describe your hiring needs: ")
    recommendations = recommend_services(user_query)
    print("\n" + recommendations + "\n")
