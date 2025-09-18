from flask import Flask, request, jsonify, render_template
import os
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "openai"])
from flask_cors import CORS
from openai import OpenAI

# Load API key and base URL from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API Key is missing. Set it in environment variables.")

OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://generativelanguage.googleapis.com/v1beta/openai/")  # Default is standard OpenAI
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gemini-2.5-flash")  # Default model

# Configure OpenAI client (supports custom base url for OpenAI-compatible APIs)
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

# Set the static folder path to the "static" folder
STATIC_FOLDER = os.path.join(os.path.dirname(__file__), "static")

# Read the context file from the static folder
CONTEXT_FILE = os.path.join(STATIC_FOLDER, "context.txt")
try:
    with open(CONTEXT_FILE, "r", encoding="utf-8") as file:
        CONTEXT_DATA = file.read()
except FileNotFoundError:
    CONTEXT_DATA = "No context available."

SYSTEM_INSTRUCTION = """
You are Hal, an AI assistant created to help farmers.
Your goal is to analyze the crop data provided in the context file and assist farmers by answering their queries and solving their problems.

Response Rules:
1. Use the context file to answer farmer-related questions.
2. Do not share any personal information.
3. Provide only the information available in the context file.
4. If not found, generate helpful answers from agricultural knowledge.
5. Keep responses simple, clear, and useful.
"""

# Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)


def translate_text(text, target_lang):
    try:
        return text
    except Exception as e:
        return f"Translation error: {str(e)}"

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message")
        target_lang = data.get("lang", "en")

        if not user_message:
            return jsonify({"error": "Message is required"}), 400

        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": f"Context Data:\n{CONTEXT_DATA}\n\nUser Query: {user_message}"}
        ]

        # Call OpenAI (or OpenAI-compatible) chat API
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.7
        )
        ai_response = response.choices[0].message.content.strip()

        if target_lang.lower() != "en":
            ai_response = translate_text(ai_response, target_lang)

        return jsonify({"response": ai_response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
