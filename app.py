from flask import Flask, request, render_template, jsonify, redirect, url_for, session, flash
import os
from openai import OpenAI
from dotenv import load_dotenv
import time
import tiktoken
import uuid
from werkzeug.security import generate_password_hash, check_password_hash

# Load environment variables from .env file
load_dotenv()

# Create OpenAI client instance
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "default_secret_key_change_in_production")

# Load the system prompt from a file
with open("prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read().strip()

# In-memory storage for user accounts (use a database in production)
users = {}

# In-memory storage for conversations and rate limiting (use a database in production)
conversations = {}
message_count = {}
message_limit = 25  # Set your desired message limit here
reset_time = 10800  # Reset counter every 3 hours (10800 seconds)

# Maximum token limit for conversation context
MAX_TOKENS = 2500  # Adjust based on your model's context window

# Initialize tokenizer for the model
# Since tiktoken doesn't recognize 'gpt-4.1-mini', we use 'o200k_base' encoding
tokenizer = tiktoken.get_encoding("o200k_base")

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
    # Redirect to login if not logged in
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("index.html", username=session["username"])

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        
        # Check if username already exists
        if username in users:
            flash("Username already exists")
            return redirect(url_for("register"))
        
        # Hash the password and store the user
        users[username] = {
            "password_hash": generate_password_hash(password),
            "created_at": time.time()
        }
        
        # Initialize conversation history for the new user
        conversations[username] = [{"role": "system", "content": system_prompt}]
        
        # Initialize message count for the new user
        message_count[username] = {'count': 0, 'timestamp': time.time()}
        
        # Log the user in
        session["username"] = username
        
        return redirect(url_for("home"))
    
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        
        # Check if user exists and password is correct
        if username in users and check_password_hash(users[username]["password_hash"], password):
            session["username"] = username
            return redirect(url_for("home"))
        else:
            flash("Invalid username or password")
            return redirect(url_for("login"))
    
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("login"))

@app.route("/chat", methods=["POST"])
def chat():
    # Check if user is logged in
    if "username" not in session:
        return jsonify({"reply": "Not authenticated"}), 401
    
    username = session["username"]
    user_input = request.json.get("message", "")
    
    # Initialize user data if not exists
    if username not in message_count:
        message_count[username] = {'count': 0, 'timestamp': time.time()}
    
    if username not in conversations:
        conversations[username] = [{"role": "system", "content": system_prompt}]
    
    # Check if the limit period has passed (reset counter)
    if time.time() - message_count[username]['timestamp'] > reset_time:
        message_count[username] = {'count': 0, 'timestamp': time.time()}

    # Check if the user has exceeded the message limit
    if message_count[username]['count'] >= message_limit:
        return jsonify({"reply": "Message limit exceeded. Please try again later."}), 403

    # Add the user message to conversation history
    conversations[username].append({"role": "user", "content": user_input})
    
    # Prepare messages for API, ensuring we stay within token limits
    messages = trim_conversation_history(conversations[username])
    
    try:
        # Make API call with conversation history
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages
        )
        
        reply = response.choices[0].message.content
        
        # Add assistant's response to conversation history
        conversations[username].append({"role": "assistant", "content": reply})
        
        # Increment message count for the user
        message_count[username]['count'] += 1
        
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
    # Check if user is logged in
    if "username" not in session:
        return jsonify({"reply": "Not authenticated"}), 401
    
    username = session["username"]
    
    if username in conversations:
        # Keep the system prompt, remove all other messages
        system_message = conversations[username][0]
        conversations[username] = [system_message]
        
    return jsonify({"status": "Conversation history reset successfully"})

def parse_flashcards(text):
    """Parse flashcard data from AI response text"""
    try:
        # Look for JSON-like structure in the response
        start_idx = text.find('{')
        end_idx = text.rfind('}') + 1
        if start_idx == -1 or end_idx == 0:
            return None
        
        import json
        flashcard_data = json.loads(text[start_idx:end_idx])
        return flashcard_data
    except:
        return None

@app.route("/generate_flashcards", methods=["POST"])
def generate_flashcards():
    if "username" not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    username = session["username"]
    topic = request.json.get("topic", "")
    
    # Add the user's request to conversation history
    conversations[username].append({
        "role": "user", 
        "content": f"Generate flashcards for the following topic: {topic}. Format the response as a JSON object with 'cards' array containing objects with 'front' and 'back' properties."
    })
    
    # Prepare messages for API
    messages = trim_conversation_history(conversations[username])
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages
        )
        
        reply = response.choices[0].message.content
        flashcard_data = parse_flashcards(reply)
        
        if flashcard_data and "cards" in flashcard_data:
            # Add assistant's response to conversation history
            conversations[username].append({"role": "assistant", "content": reply})
            return jsonify({"flashcards": flashcard_data["cards"]})
        else:
            return jsonify({"error": "Failed to generate flashcards"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)