import os
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AIMessage
from dotenv import load_dotenv
from pydantic import BaseModel, EmailStr
from typing import List, Literal, Optional
from fastapi import FastAPI
import hashlib

app = FastAPI()
checkpointer = InMemorySaver()

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

def normalize_email(email: str) -> str:
    """
    Normalizes the email so the same logical email always maps to the same string.
    You can extend this (e.g., remove dots for gmail local part) depending on rules.
    """
    return email.strip().lower()

def hash_email_sha256(email: str, *, as_hex: bool = True) -> str:
    """
    Deterministic SHA-256 hash of the normalized email.
    Returns hex string by default (set as_hex=False to get raw bytes).
    """
    norm = normalize_email(email)
    digest = hashlib.sha256(norm.encode("utf-8")).digest()
    return digest.hex() if as_hex else digest

# Prompts for each model
prompts = {
    "normal": """
You are an AI Recruitment Assistant - YOUR NAME IS DelCap Agent.

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
  If it's completely out of scope, say:
  "I'm here to help you with recruitment. Do you have any hiring needs at the moment?"

Your style:
- Casual and approachable but subtly steering toward recruitment.
- Use small talk sparingly, then return to recruitment questions.
- Ask gentle, open-ended follow-ups to discover user needs.
""",
    "sales": """
You are an AI Sales Assistant for a recruitment agency - YOUR NAME IS DelCap Agent.
Do not refuse for any individual looking also. properly present. you are recruitment agency your job is to recommend according to these profiles. do not refuse for individual. and please print the package firstg and then proceed to pitch.
Our agency offers the following 10 fixed recruitment service packages (choose ONLY from this list, never invent new ones):
ALWAYS START OFF BY MENTIONING THE PACKAGE NAME IN HIGHLIGHTED BOLD MENTIONS. THAT SHOULD ALWAYS BE THE FIRST SENTENCE.
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
Other (not industry-specific) Hiring Pack - ₹6,00,000

The pitch must include:
ALWAYS START OFF BY MENTIONING THE PACKAGE NAME IN HIGHLIGHTED BOLD MENTIONS. THAT SHOULD ALWAYS BE THE FIRST SENTENCE.
A clear subject line.
A brief opening acknowledging the user's need.
Why the chosen package fits their case (1-2 sentences).
Key benefits of the package.
A strong, direct call to action.

Polished, business-professional tone (avoid fluff or technical jargon).
you first need to suggest one of the packages. choose a package first and then try to give a sales pitch as to why user should choose it.
You will write "Dear User" and end with "recruitment assistant".
you're doing well, we just need to make sure the answer looks a little less ai generated. make it looks more natural like how a human would pitch.
👉 Do NOT output excessive details, tables, or JSON — only a clean, real-world business email proposal.
ALWAYS START OFF BY MENTIONING THE PACKAGE NAME IN HIGHLIGHTED BOLD MENTIONS. THAT SHOULD ALWAYS BE THE FIRST SENTENCE.
""",
    "data_extraction": """
You are an AI Sales Assistant for a recruitment agency - YOUR NAME IS DelCap Agent.
Your job is to carefully extract structured data from the user's hiring request.

Always return a clean JSON object with the following fields [THE RETURNED VALUE SHOULD FOLLOW THE RESPONSE FORMAT GIVEN TO YOU - DONT RETURN ANYTHING EXCEPT THE JSON, NOT EVEN ```json]:
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
- Only return JSON (no text outside JSON, NOT EVEN MARKDOWN TEXT, RETURN PYTHON DICTIONARY).
- If a field is missing, return null.
""",
    "intent": """
You are an intent classification agent for a recruiting agency  - YOUR NAME IS DelCap Agent.
Your task is to analyze the user's free-text input (which may include surrounding context or questions).

You must decide whether the conversation should:
- **Trigger the recommendation engine** → if the JSON and/or user input clearly relate to recruitment needs, such as hiring requirements, open roles, urgency, industry or location-based hiring, or service inquiries about recruitment.  
- **Trigger the normal conversation engine** → if the user is only making small talk, asking about unrelated topics, or the input is not actionable for recruitment recommendations.

### Output Format
You must respond ONLY in valid JSON that strictly follows this schema:

{
  "intent": "normal" | "recruitment"
}

- Use `"recruitment"` if recommending recruitment services is appropriate.  
- Use `"normal"` otherwise.  
- Do not add extra fields or text outside the JSON.  
- Be decisive — always choose exactly one intent.

# IMPORTANT: TRY TENDING TOWARDS RECRUITMENT MORE THAN NORMAL, GIVE MORE PRIORITY TO RECRUITMENT. ONLY SUGGEST NORMAL WHEN RECRUITMENT IS NOT POSSIBLE FROM THE CONTEXT AT ALL.
"""
}

class JobResponseModel(BaseModel):
    industry: Optional[str] = None
    location: Optional[str] = None
    roles: List[str]
    count: Optional[int] = None
    urgency: bool

class IntentResponseModel(BaseModel):
    intent: Literal["normal", "recruitment"]

# Initialize Gemini model (2.5 Flash)
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_api_key
)

normal_conversation_agent = create_react_agent(
    model=model,
    tools=[],
    prompt=prompts["normal"],
    checkpointer=checkpointer
)

recommendation_agent = create_react_agent(
    model=model,
    tools=[],
    prompt=prompts["sales"],
    checkpointer=checkpointer
)

data_extraction_agent = create_react_agent(
    model=model,
    tools=[],
    prompt=prompts["data_extraction"],
    response_format=JobResponseModel,
    checkpointer=checkpointer
)

intent_classification_agent = create_react_agent(
    model=model,
    tools=[],
    prompt=prompts["intent"],
    response_format=IntentResponseModel,
    checkpointer=checkpointer
)

def classify_from_data(extracted, config, user_message) -> str:
    op = user_message
    res = intent_classification_agent.invoke({"messages": [{"role": "user", "content": op}]}, config)

    structured = res["structured_response"]

    return structured.intent


def extract_data(user_message: str, config):
    """Send user message to the agent and return parsed JSON output."""
    result = data_extraction_agent.invoke({"messages": [{"role": "user", "content": user_message}]}, config)
    structured = result["structured_response"]

    return structured

def recommend_services(user_message: str, config) -> str:
    result = recommendation_agent.invoke({"messages": [{"role": "user", "content": user_message}]}, config)

    if hasattr(result, "content"):
        return result.content.strip()
    elif isinstance(result, dict) and "output" in result:
        return result["output"].strip()
    elif isinstance(result, dict) and "messages" in result:
        return result["messages"][-1].content.strip()
    return str(result).strip()

@app.get("/")
async def root():
    return {"message": "Welcome to Recruitment Agent LLM"}

class ConverseRequest(BaseModel):
    email: EmailStr
    message: str
    
@app.post("/converse")
async def converse(request: ConverseRequest):
    hashed_email = hash_email_sha256(normalize_email(request.email))
    config = {"configurable": {"thread_id": hashed_email}}

    # Step 1: Extract data
    extracted_json = extract_data(request.message, config)
    print(extracted_json)

    # Step 2: Classify intent
    intent = classify_from_data(extracted_json, config, request.message)
    print("Intent", intent)

    # Step 3: Route to correct agent
    if intent == "normal":
        response = normal_conversation_agent.invoke(
            {"messages": [{"role": "user", "content": request.message}]},
            config
        )
        return {
            "intent": "normal",
            "response": response["messages"][-1].content
        }

    elif intent == "recruitment":
        recommendations = recommend_services(request.message, config)
        return {
            "intent": "recruitment",
            "extracted_data": extracted_json,
            "recommendations": recommendations
        }

    # fallback if nothing matches
    return {
        "intent": intent,
        "response": "No valid route found."
    }
