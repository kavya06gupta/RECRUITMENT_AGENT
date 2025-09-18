DelCap Recruitment Agent

DelCap Agent is an AI-powered recruitment assistant that blends natural conversation with intelligent hiring insights.

It can:

Chat naturally with users while gently steering toward recruitment topics.

Detect when a conversation involves hiring needs.

Extract key details such as roles, location, count, urgency, and industry.

Recommend from a fixed set of recruitment service packages and deliver a professional sales pitch.

Maintain context across conversations using secure, hashed email IDs.

Try it here 👉 Live on Render

✨ What It Does

Normal Conversations
Handles greetings and small talk naturally, while keeping recruitment in focus.

Recruitment Conversations
Detects hiring needs, extracts structured data, and recommends the most suitable service package with a polished sales pitch.

Data Extraction
Turns free-form text into structured fields like industry, location, job roles, hiring count, and urgency.

Intent Classification
Decides whether the conversation should stay casual or move to recruitment services.

🚀 Running Locally
1. Clone the Repository
git clone https://github.com/your-username/delcap-recruitment-agent.git
cd delcap-recruitment-agent

2. Create Virtual Environment
python -m venv venv
# Activate it
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

3. Install Requirements
pip install -r requirements.txt

4. Run the App
fastapi dev app.py


The app will start locally on http://127.0.0.1:8000

🌐 Deployment

This project is deployed on Render(it is a little slow due to this) and publicly accessible here:

👉 https://recruitment-agent-1.onrender.com/

🛠 Tech Stack

FastAPI – Web framework

LangGraph – Agent orchestration

LangChain Google GenAI – LLM integration

Pydantic – Data validation

dotenv – Environment variables

📄 License

MIT License © 2025 — DelCap
