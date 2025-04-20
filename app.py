from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
import os
import requests
from flask_cors import CORS
from googletrans import Translator

# Load API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Gemini API Key is missing. Set it in environment variables.")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(model_name="gemini-2.0-flash")

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

        prompt = [
            SYSTEM_INSTRUCTION,
            f"Context Data:\n{CONTEXT_DATA}",
            f"\n\nUser Query: {user_message}"
        ]
        ai_response = model.generate_content(prompt).text.strip()

        if target_lang.lower() != "en":
            ai_response = translate_text(ai_response, target_lang)

        return jsonify({"response": ai_response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# For local testing, you can use the following:
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
