from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS
from googletrans import Translator
import litellm # MODIFIED: Import litellm instead of OpenAI

# Load API key, base URL, and model from environment variables
# LiteLLM can automatically read OPENAI_API_KEY, but we'll read it explicitly for clarity
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("API Key is missing. Set OPENAI_API_KEY or GEMINI_API_KEY in environment variables.")

# Note: For litellm, the parameter is 'api_base' not 'base_url'
API_BASE = os.getenv("OPENAI_API_BASE", "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gemini-1.5-flash") # Default model

# REMOVED: No need to instantiate a client with litellm
# client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

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

        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": f"Context Data:\n{CONTEXT_DATA}\n\nUser Query: {user_message}"}
        ]

        # MODIFIED: Call litellm.completion directly
        # Pass the model, messages, api_key, and api_base here
        response = litellm.completion(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.7,
            api_key=API_KEY,
            api_base=API_BASE
        )
        
        # The response structure is the same as OpenAI's, so this part doesn't change
        ai_response = response.choices[0].message.content.strip()

        if target_lang.lower() != "en":
            ai_response = translate_text(ai_response, target_lang)

        return jsonify({"response": ai_response})
    except Exception as e:
        # LiteLLM can raise specific exceptions, but catching the general one is fine
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
