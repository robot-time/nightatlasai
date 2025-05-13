from flask import Flask, request, render_template, jsonify
import os
from openai import OpenAI
from dotenv import load_dotenv
import time
import tiktoken

# Load .env locally
load_dotenv()

# Create OpenAI client instance
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)

# Load the system prompt from a file
with open("prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read().strip()

# In-memory storage for conversations and rate limiting (use a database in production)
conversations = {}
message_count = {}
message_limit = 25  # Set your desired message limit here
reset_time = 10800  # Reset counter every 3 hours (10800 seconds)

# Maximum token limit for conversation context
MAX_TOKENS = 2500  # Adjust based on your model's context window

# Initialize tokenizer for the model
tokenizer = tiktoken.encoding_for_model("gpt-4.1-mini")

def num_tokens_from_messages(messages):
    """Calculate the number of tokens in a message list"""
    num_tokens = 0
    for message in messages:
        # Add tokens for message role and content
        num_tokens += 4  # Every message has a 4 token overhead
        for key, value in message.items():
            num_tokens += len(tokenizer.encode(value))
    num_tokens += 2  # Add 2 for the final assistant message role overhead
    return num_tokens

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_id = request.json.get("user_id", "default_user")  # Unique user identifier
    user_input = request.json.get("message", "")
    
    # Initialize user data if not exists
    if user_id not in message_count:
        message_count[user_id] = {'count': 0, 'timestamp': time.time()}
    
    if user_id not in conversations:
        conversations[user_id] = [{"role": "system", "content": system_prompt}]
    
    # Check if the limit period has passed (reset counter)
    if time.time() - message_count[user_id]['timestamp'] > reset_time:
        message_count[user_id] = {'count': 0, 'timestamp': time.time()}

    # Check if the user has exceeded the message limit
    if message_count[user_id]['count'] >= message_limit:
        return jsonify({"reply": "Message limit exceeded. Please try again later."}), 403

    # Add the user message to conversation history
    conversations[user_id].append({"role": "user", "content": user_input})
    
    # Prepare messages for API, ensuring we stay within token limits
    messages = trim_conversation_history(conversations[user_id])
    
    try:
        # Make API call with conversation history
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        
        reply = response.choices[0].message.content
        
        # Add assistant's response to conversation history
        conversations[user_id].append({"role": "assistant", "content": reply})
        
        # Increment message count for the user
        message_count[user_id]['count'] += 1
        
        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"}), 500

def trim_conversation_history(messages):
    """Trim conversation history to fit within token limits"""
    # Always keep the system prompt (first message)
    system_message = messages[0]
    
    # Start with just the system message
    trimmed_messages = [system_message]
    current_tokens = num_tokens_from_messages(trimmed_messages)
    
    # Add as many recent messages as possible, starting from the most recent
    for message in reversed(messages[1:]):
        message_tokens = num_tokens_from_messages([message])
        
        if current_tokens + message_tokens <= MAX_TOKENS:
            trimmed_messages.insert(1, message)  # Insert after system message
            current_tokens += message_tokens
        else:
            # If we can't add any more messages, break
            break
    
    return trimmed_messages

@app.route("/reset", methods=["POST"])
def reset_conversation():
    user_id = request.json.get("user_id", "default_user")
    
    if user_id in conversations:
        # Keep the system prompt, remove all other messages
        system_message = conversations[user_id][0]
        conversations[user_id] = [system_message]
        
    return jsonify({"status": "Conversation history reset successfully"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)