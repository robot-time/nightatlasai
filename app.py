from flask import Flask, request, render_template, jsonify, redirect, url_for, session, flash
import os
from openai import OpenAI
from dotenv import load_dotenv
import time
import tiktoken
import uuid
from werkzeug.security import generate_password_hash, check_password_hash
import json

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
reset_time = 14400  # Reset counter every 4 hours (14400 seconds)

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

def get_remaining_messages(username):
    """Get the number of remaining messages and time until reset"""
    if username not in message_count:
        return message_limit, 0
    
    count_data = message_count[username]
    time_elapsed = time.time() - count_data['timestamp']
    
    if time_elapsed >= reset_time:
        return message_limit, 0
    
    remaining = message_limit - count_data['count']
    time_until_reset = reset_time - time_elapsed
    
    return remaining, time_until_reset

def format_time_remaining(seconds):
    """Format seconds into a human-readable time string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    
    if hours > 0:
        return f"{hours} hour{'s' if hours != 1 else ''} and {minutes} minute{'s' if minutes != 1 else ''}"
    return f"{minutes} minute{'s' if minutes != 1 else ''}"

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
    remaining, time_until_reset = get_remaining_messages(username)
    if remaining <= 0:
        time_str = format_time_remaining(time_until_reset)
        return jsonify({
            "reply": f"You've reached your message limit. Please try again in {time_str}.",
            "error": "rate_limit",
            "time_until_reset": time_until_reset
        }), 429

    # Check if this is a flashcard request
    if is_flashcard_request(user_input):
        # Extract topic from the message
        topic = user_input.replace("flashcard", "").replace("flash card", "").strip()
        if topic:
            try:
                response = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[{
                        "role": "user",
                        "content": f"Extract the main topic from this flashcard request: {user_input}"
                    }]
                )
                topic = response.choices[0].message.content.strip()
            except:
                pass
        
        return jsonify({
            "reply": f"I'll create some flashcards about {topic} for you!",
            "flashcard_request": True,
            "topic": topic,
            "remaining_messages": remaining - 1
        })
    
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
        
        return jsonify({
            "reply": reply,
            "remaining_messages": remaining - 1
        })

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
        
        json_str = text[start_idx:end_idx]
        # Clean up the JSON string
        json_str = json_str.replace("'", '"')  # Replace single quotes with double quotes
        json_str = json_str.replace("\n", " ")  # Remove newlines
        json_str = json_str.replace("\\", "\\\\")  # Escape backslashes
        
        flashcard_data = json.loads(json_str)
        return flashcard_data
    except Exception as e:
        print(f"Parse Error: {str(e)}")  # Log the error
        return None

def is_flashcard_request(text):
    """Check if the message is requesting flashcards"""
    # Common phrases that might indicate a flashcard request
    flashcard_indicators = [
        "flashcard", "flash card", "study cards", "review cards",
        "quiz cards", "memory cards", "learning cards"
    ]
    
    # Check if any indicator is in the text
    return any(indicator in text.lower() for indicator in flashcard_indicators)

@app.route("/generate_flashcards", methods=["POST"])
def generate_flashcards():
    if "username" not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    try:
        username = session["username"]
        data = request.get_json()
        if not data or "topic" not in data:
            return jsonify({"error": "No topic provided"}), 400
            
        topic = data["topic"]
        
        # Create a specific prompt for flashcard generation
        flashcard_prompt = f"""Create a set of educational flashcards about: {topic}

Requirements:
1. Generate 5-7 flashcards covering key concepts
2. Each card should have a clear question on the front and a detailed answer on the back
3. Format the response as a JSON object with a 'cards' array
4. Each card should have 'front' and 'back' properties
5. Keep questions concise and answers informative but not too long

Example format:
{{
  "cards": [
    {{
      "front": "What is X?",
      "back": "X is..."
    }},
    {{
      "front": "Define Y",
      "back": "Y is..."
    }}
  ]
}}

Please provide the flashcards in this exact JSON format."""

        try:
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{
                    "role": "user",
                    "content": flashcard_prompt
                }],
                temperature=0.7,
                max_tokens=1000
            )
            
            reply = response.choices[0].message.content.strip()
            
            # Try to parse the JSON response
            try:
                # First, try to find JSON in the response
                start_idx = reply.find('{')
                end_idx = reply.rfind('}') + 1
                if start_idx == -1 or end_idx == 0:
                    return jsonify({"error": "Invalid response format"}), 400
                
                json_str = reply[start_idx:end_idx]
                # Clean up the JSON string
                json_str = json_str.replace("'", '"')  # Replace single quotes with double quotes
                json_str = json_str.replace("\n", " ")  # Remove newlines
                json_str = json_str.replace("\\", "\\\\")  # Escape backslashes
                
                flashcard_data = json.loads(json_str)
                
                # Validate the flashcard data structure
                if not isinstance(flashcard_data, dict) or "cards" not in flashcard_data:
                    return jsonify({"error": "Invalid flashcard format"}), 400
                
                if not isinstance(flashcard_data["cards"], list):
                    return jsonify({"error": "Cards must be an array"}), 400
                
                # Validate each card
                for card in flashcard_data["cards"]:
                    if not isinstance(card, dict) or "front" not in card or "back" not in card:
                        return jsonify({"error": "Invalid card format"}), 400
                    if not isinstance(card["front"], str) or not isinstance(card["back"], str):
                        return jsonify({"error": "Card content must be strings"}), 400
                
                return jsonify({
                    "flashcards": flashcard_data["cards"],
                    "topic": topic
                })
                
            except json.JSONDecodeError as e:
                print(f"JSON Parse Error: {str(e)}")
                print(f"Raw response: {reply}")
                return jsonify({"error": "Failed to parse flashcard data"}), 400
                
        except Exception as e:
            print(f"OpenAI API Error: {str(e)}")
            return jsonify({"error": "Failed to generate flashcards"}), 500

    except Exception as e:
        print(f"General Error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)