from flask import Flask, request, render_template, jsonify
import os
from openai import OpenAI
from dotenv import load_dotenv
import time

# Load .env locally
load_dotenv()

# Create OpenAI client instance
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)

# Load the system prompt from a file
with open("prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read().strip()

# In-memory dictionary to store message counts (for demo purposes, use a persistent database in production)
message_count = {}
message_limit = 25  # Set your desired message limit here
reset_time = 10800  # Reset counter every 24 hours (86400 seconds)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_id = request.json.get("user_id")  # Assume each user has a unique 'user_id'
    user_input = request.json.get("message", "")

    if user_id not in message_count:
        message_count[user_id] = {'count': 0, 'timestamp': time.time()}

    # Check if the limit period has passed (reset every 24 hours)
    if time.time() - message_count[user_id]['timestamp'] > reset_time:
        message_count[user_id] = {'count': 0, 'timestamp': time.time()}

    # Check if the user has exceeded the message limit
    if message_count[user_id]['count'] >= message_limit:
        return jsonify({"reply": "Message limit exceeded, please try again tomorrow."}), 403

    # Otherwise, proceed with the OpenAI API call
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        reply = response.choices[0].message.content
        
        # Increment message count for the user
        message_count[user_id]['count'] += 1
        
        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
