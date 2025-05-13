from flask import Flask, request, render_template, jsonify, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

# Load .env locally
load_dotenv()

# Create OpenAI client instance
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Flask app, SQLAlchemy, and LoginManager
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)  # Secret key for session management
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'  # SQLite database (change for production)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"  # Redirect to login if not authenticated

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

# Load user for login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# In-memory dictionary to store message counts (for demo purposes, use a persistent database in production)
message_count = {}
message_limit = 25  # Set your desired message limit here
reset_time = 86400  # Reset counter every 24 hours (86400 seconds)

# Route for home page
@app.route("/")
def home():
    return render_template("index.html")

# Registration route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password, method='sha256')
        
        # Check if username already exists
        user = User.query.filter_by(username=username).first()
        if user:
            return "Username already exists!"
        
        # Create a new user
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        login_user(new_user)
        return redirect(url_for('dashboard'))
    
    return render_template('register.html')

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            return "Invalid credentials, please try again."

    return render_template('login.html')

# Logout route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

# Dashboard route (Only accessible for logged-in users)
@app.route('/dashboard')
@login_required
def dashboard():
    return f"Welcome to your dashboard, {current_user.username}!"

# Chat route (only accessible to logged-in users)
@app.route("/chat", methods=["POST"])
@login_required
def chat():
    user_id = current_user.id  # Use the logged-in user's ID
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
        {"role": "system", "content": "Your system prompt here."},
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

# Create all tables before the first request (needed for Flask-Migrate)
@app.before_first_request
def create_tables():
    db.create_all()

# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
