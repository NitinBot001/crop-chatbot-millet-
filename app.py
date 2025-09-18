from flask import Flask, request, jsonify, render_template
from openai import OpenAI
import os
import requests
from flask_cors import CORS
from googletrans import Translator

# Load API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API Key is missing. Set it in environment variables.")

# Configure OpenAI
base_url = os.getenv("BASE_URL","https://generativelanguage.googleapis.com/v1beta/openai/")
ai_model = os.getenv("MODEL","gemini-2.5-flash")
client = OpenAI(base_url=base_url,api_key=OPENAI_API_KEY)

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

# Translator instance
translator = Translator()

def translate_text(text, target_lang):
    try:
        translated = translator.translate(text, dest=target_lang)
        return translated.text
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

        # Create messages for OpenAI chat completion
        messages = [
            {"role": "system", "content": f"{SYSTEM_INSTRUCTION}\n\nContext Data:\n{CONTEXT_DATA}"},
            {"role": "user", "content": user_message}
        ]

        # Call OpenAI API
        response = client.chat.completions.create(
            model=ai_model,  # or "gpt-3.5-turbo" for cheaper option, or "gpt-4" for better quality
            messages=messages,
            temperature=0.7,
            max_tokens=10000
        )
        
        ai_response = response.choices[0].message.content.strip()

        if target_lang.lower() != "en":
            ai_response = translate_text(ai_response, target_lang)

        return jsonify({"response": ai_response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# For local testing, you can use the following:
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
