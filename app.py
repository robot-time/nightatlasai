from flask import Flask, request, render_template, jsonify
import openai
import os

from dotenv import load_dotenv
load_dotenv()  # Only for local development

openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

with open("prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read().strip()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        reply = response["choices"][0]["message"]["content"]
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"reply": f"Error: {e}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
