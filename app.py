from flask import Flask, request, render_template, jsonify, redirect, url_for, session, flash
import os
from openai import OpenAI
from dotenv import load_dotenv
import time
import tiktoken
import uuid
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import json
import PyPDF2
import re
import firebase_auth

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("WARNING: OPENAI_API_KEY not found in environment variables. AI features will not work.")

try:
    # Create OpenAI client instance
    client = OpenAI(api_key=openai_api_key)
    print("OpenAI client initialized successfully")
except Exception as e:
    print(f"ERROR initializing OpenAI client: {str(e)}")
    client = None

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "default_secret_key_change_in_production")

# Configure file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create upload directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load the system prompt from a file
with open("prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read().strip()

# In-memory storage for conversations and rate limiting (use a database in production)
conversations = {}
message_count = {}
message_limit = 25  # Set your desired message limit here
reset_time = 14400  # Reset counter every 4 hours (14400 seconds)

# In-memory storage for uploaded PDFs and their extracted content
pdf_storage = {}

# Maximum token limit for conversation context
MAX_TOKENS = 2500  # Adjust based on your model's context window

# Initialize tokenizer for the model
# Since tiktoken doesn't recognize 'gpt-4.1-mini', we use 'o200k_base' encoding
tokenizer = tiktoken.get_encoding("o200k_base")

# User session helper functions
def is_logged_in():
    """Check if the user is logged in"""
    return 'user_id' in session and 'email' in session

def get_user_id():
    """Get the current user's ID"""
    if is_logged_in():
        return session['user_id']
    return None

def get_user_email():
    """Get the current user's email"""
    if is_logged_in():
        return session['email']
    return None

def get_username():
    """Get the current username"""
    if is_logged_in():
        return session.get('username', '')
    return None

def extract_text_from_pdf(file_path):
    """Extract text content from a PDF file"""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return None

def extract_tables_from_text(text):
    """
    Attempts to identify and extract tables from text by looking for structured patterns
    This is a simplified approach and may need refinement based on specific rubric formats
    """
    # Store potential table rows
    potential_table_rows = []
    
    # Try different pattern matching approaches
    
    # Approach 1: Look for grade band patterns (e.g., "A: 90-100" or "Excellent: Clear evidence of...")
    grade_pattern = r'(?:^|\n)([A-F][+\-]?|[0-9]{1,2}[\-][0-9]{1,2}|Excellent|Good|Satisfactory|Pass|Fail|[A-Z][a-z]*)\s*[:|-]\s*(.+?)(?=\n[A-F][+\-]?|[0-9]{1,2}[\-][0-9]{1,2}|Excellent|Good|Satisfactory|Pass|Fail|[A-Z][a-z]*\s*[:|-]|\Z)'
    matches = re.finditer(grade_pattern, text, re.DOTALL)
    
    for match in matches:
        grade = match.group(1).strip()
        description = match.group(2).strip()
        potential_table_rows.append({"grade": grade, "description": description})
    
    # Approach 2: Look for tabular data with grade bands in separate columns
    # This pattern tries to identify rows that look like tables with multiple columns
    if not potential_table_rows:
        # Look for lines with multiple tab or multiple space separations
        lines = text.split('\n')
        table_start = False
        current_table = []
        
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue
                
            # Check if this might be a table row (contains multiple tabs or 3+ spaces in sequence)
            if '\t' in line or '   ' in line:
                # Could be a table row
                if not table_start:
                    table_start = True
                    
                # Split by tabs or multiple spaces
                if '\t' in line:
                    columns = [col.strip() for col in line.split('\t')]
                else:
                    columns = [col.strip() for col in re.split(r'\s{3,}', line)]
                    
                # If we have at least 2 columns and first looks like a grade marker
                if len(columns) >= 2:
                    first_col = columns[0].strip()
                    # Check if first column looks like a grade marker
                    if (re.match(r'^[A-F][+\-]?$', first_col) or 
                        re.match(r'^[0-9]{1,2}[\-][0-9]{1,2}$', first_col) or
                        first_col in ['Excellent', 'Good', 'Satisfactory', 'Pass', 'Fail']):
                        
                        # Join the rest of the columns as the description
                        description = ' '.join(columns[1:]).strip()
                        potential_table_rows.append({"grade": first_col, "description": description})
            else:
                # Reset table detection if we've already started a table
                table_start = False
    
    # Approach 3: Look for numbered criteria with descriptions
    if not potential_table_rows:
        numbered_pattern = r'(?:^|\n)(\d+\.)\s+(.+?)(?=\n\d+\.|\Z)'
        matches = re.finditer(numbered_pattern, text, re.DOTALL)
        
        for match in matches:
            grade = match.group(1).strip()
            description = match.group(2).strip()
            potential_table_rows.append({"grade": grade, "description": description})
    
    return potential_table_rows

def analyze_pdf_content(file_path, file_type="generic"):
    """
    Analyze the PDF content based on the specified file type
    file_type options: "generic", "rubric"
    """
    text = extract_text_from_pdf(file_path)
    if not text:
        return {"error": "Failed to extract text from PDF"}
    
    # Special handling for rubrics
    if file_type == "rubric":
        tables = extract_tables_from_text(text)
        
        # If we couldn't extract tables with our pattern-based approach, try using AI
        if not tables:
            tables = extract_rubric_with_ai(text)
            
        if tables:
            return {
                "text": text,
                "tables": tables,
                "summary": f"Extracted {len(tables)} grading criteria from rubric"
            }
    
    # For generic PDFs, just return the text
    return {
        "text": text,
        "summary": f"Extracted {len(text.split())} words from PDF"
    }

def extract_rubric_with_ai(text):
    """
    Use OpenAI to extract rubric data when pattern-based approaches don't work
    """
    try:
        # Create a prompt for the AI to extract rubric data
        prompt = f"""
        Extract grading criteria from the following rubric text. 
        For each grade band or criterion, identify:
        1. The grade label/band (e.g., A, B, 90-100, Excellent, etc.)
        2. The description or requirements for that grade

        Format the output as a JSON array of objects with 'grade' and 'description' fields.
        
        Text:
        {text[:2000]}  # Limit text length to avoid exceeding token limits
        """
        
        response = client.chat.completions.create(
            model="gpt-4.1-mini",  # Using the same model as the chat
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3  # Lower temperature for more deterministic output
        )
        
        response_text = response.choices[0].message.content
        
        # Try to parse the JSON response
        try:
            data = json.loads(response_text)
            if isinstance(data, dict) and "criteria" in data:
                return data["criteria"]
            elif "rows" in data:
                return data["rows"]
            elif isinstance(data, list):
                return data
            else:
                # Try to extract an array from the response
                for key, value in data.items():
                    if isinstance(value, list) and len(value) > 0:
                        if isinstance(value[0], dict) and "grade" in value[0] and "description" in value[0]:
                            return value
                
                # If we can't find a properly structured array, create one from the data
                result = []
                for key, value in data.items():
                    if isinstance(value, str):
                        result.append({"grade": key, "description": value})
                return result
                
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract data with regex
            print(f"AI returned non-JSON output: {response_text}")
            return []
            
    except Exception as e:
        print(f"Error using AI to extract rubric: {str(e)}")
        return []

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

def get_remaining_messages(user_id):
    """Get the number of remaining messages and time until reset"""
    # Try to get user data from Firebase first
    user_data = firebase_auth.get_user_data(user_id)
    
    # If we have user data from Firebase, use those limits
    if user_data:
        message_limit = user_data.get('message_limit', 25)
        reset_time_seconds = user_data.get('reset_time', 14400)  # Default 4 hours
        current_count = user_data.get('message_count', 0)
        last_reset = user_data.get('last_reset', 0)
        
        # If last_reset is a Firestore timestamp, convert to epoch seconds
        if hasattr(last_reset, 'seconds'):
            last_reset = last_reset.seconds
            
        # Check if it's time to reset the counter
        time_elapsed = time.time() - last_reset
        if time_elapsed >= reset_time_seconds:
            return message_limit, 0
        
        remaining = message_limit - current_count
        time_until_reset = reset_time_seconds - time_elapsed
        
        return remaining, time_until_reset
    
    # Fallback to in-memory tracking
    if user_id not in message_count:
        return message_limit, 0
    
    count_data = message_count[user_id]
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
    if not is_logged_in():
        return redirect(url_for("login"))
    return render_template("index.html", username=get_username())

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        username = request.form["username"]
        
        # Register the user with Firebase
        result = firebase_auth.register_user(email, password, username)
        
        if result['success']:
            # Store user info in session
            user = result['user']
            session['user_id'] = user['localId']
            session['email'] = email
            session['username'] = username
            session['token'] = user['idToken']
            
            # Initialize conversation history for the new user
            user_id = user['localId']
            if user_id not in conversations:
                conversations[user_id] = [{"role": "system", "content": system_prompt}]
            
            return redirect(url_for("home"))
        else:
            # Registration failed
            flash(result['message'])
            return redirect(url_for("register"))
    
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        
        # Login the user with Firebase
        result = firebase_auth.login_user(email, password)
        
        if result['success']:
            # Store user info in session
            user = result['user']
            user_data = result.get('user_data', {})
            
            session['user_id'] = user['localId']
            session['email'] = email
            session['token'] = user['idToken']
            
            # Get username from user data or use email as fallback
            username = user_data.get('username', email.split('@')[0])
            session['username'] = username
            
            # Initialize conversation history if needed
            user_id = user['localId']
            if user_id not in conversations:
                conversations[user_id] = [{"role": "system", "content": system_prompt}]
            
            return redirect(url_for("home"))
        else:
            # Login failed
            flash(result['message'])
            return redirect(url_for("login"))
    
    return render_template("login.html")

@app.route("/logout")
def logout():
    # Clear session data
    session.clear()
    return redirect(url_for("login"))

@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    """Handle PDF uploads and process them"""
    if not is_logged_in():
        return jsonify({"error": "Not authenticated"}), 401
    
    user_id = get_user_id()
    username = get_username()
    
    # Check if the post request has the file part
    if 'pdf_file' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['pdf_file']
    
    # If user does not select file, browser also submits an empty part without filename
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file and allowed_file(file.filename):
        # Secure the filename and save the file
        filename = secure_filename(file.filename)
        file_id = str(uuid.uuid4())
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}_{filename}")
        file.save(file_path)
        
        # Get the file type from form data
        file_type = request.form.get('file_type', 'generic')
        
        # Process the PDF based on its type
        pdf_data = analyze_pdf_content(file_path, file_type)
        
        # Store the processed data
        # Use in-memory storage for current session
        if username not in pdf_storage:
            pdf_storage[username] = {}
            
        pdf_storage[username][file_id] = {
            "filename": filename,
            "file_path": file_path,
            "file_type": file_type,
            "processed_data": pdf_data,
            "upload_time": time.time()
        }
        
        # Also save to Firestore for persistence
        pdf_data_for_firebase = {
            "filename": filename,
            "file_path": file_path,  # Note: This is a local path, consider using Firebase Storage for the actual file
            "file_type": file_type,
            "processed_data": pdf_data,
            "upload_time": time.time()
        }
        firebase_auth.save_pdf_data(user_id, file_id, pdf_data_for_firebase)
        
        # Add context to the conversation for this PDF
        user_id = get_user_id()
        if user_id in conversations:
            if file_type == "rubric":
                # Format rubric information for the AI
                rubric_info = "I've uploaded a grading rubric with the following criteria:\n\n"
                for item in pdf_data.get("tables", []):
                    rubric_info += f"- {item['grade']}: {item['description']}\n"
                
                # Add this as a user message in the conversation
                conversations[user_id].append({"role": "user", "content": rubric_info})
                print(f"Added rubric info to conversation: {rubric_info[:100]}...")
                
                # Add a system message to guide the AI
                system_guidance = ("Remember to reference this rubric when providing feedback or "
                                  "guidance on assignments. Consider the specific requirements for "
                                  "each grade band when answering questions about assessment criteria.")
                conversations[user_id].append({"role": "system", "content": system_guidance})
            else:
                # For generic PDFs - handle large documents by chunking
                full_text = pdf_data.get("text", "")
                
                # Add a basic summary
                pdf_context = f"I've uploaded a PDF document named '{filename}'. "
                pdf_context += f"It contains approximately {len(full_text.split())} words."
                
                # Print debug info
                print(f"PDF Content Preview: {full_text[:200]}...")
                print(f"Adding PDF context to conversation: {pdf_context}")
                
                conversations[user_id].append({"role": "user", "content": pdf_context})
                
                # For larger documents, summarize content to avoid token limits
                if len(full_text.split()) > 1000:
                    try:
                        # Use the API to generate a summary of the PDF content
                        summary_prompt = f"Please summarize the key points from this document in 2-3 paragraphs:\n\n{full_text[:3000]}..."
                        
                        summary_response = client.chat.completions.create(
                            model="gpt-4.1-mini",
                            messages=[{"role": "user", "content": summary_prompt}],
                            temperature=0.3
                        )
                        
                        summary = summary_response.choices[0].message.content
                        print(f"Generated PDF summary: {summary[:100]}...")
                        
                        # Add the summary to the conversation
                        system_note = "I've analyzed the document and here's a summary of its content:"
                        conversations[user_id].append({"role": "system", "content": system_note})
                        conversations[user_id].append({"role": "assistant", "content": summary})
                    except Exception as e:
                        print(f"Error generating PDF summary: {str(e)}")
                else:
                    # For smaller documents, add the full content
                    pdf_content_msg = f"The content of the document is:\n\n{full_text}"
                    conversations[user_id].append({"role": "user", "content": pdf_content_msg})
                    print(f"Added full PDF content to conversation (length: {len(full_text)})")
        
        return jsonify({
            "success": True,
            "file_id": file_id,
            "filename": filename,
            "summary": pdf_data.get("summary", "PDF processed successfully"),
            "file_type": file_type
        })
    
    return jsonify({"error": "File type not allowed"}), 400

@app.route("/pdf_content/<file_id>", methods=["GET"])
def get_pdf_content(file_id):
    """Retrieve processed content of a specific PDF"""
    if not is_logged_in():
        return jsonify({"error": "Not authenticated"}), 401
    
    username = get_username()
    
    if username not in pdf_storage or file_id not in pdf_storage[username]:
        return jsonify({"error": "PDF not found"}), 404
    
    pdf_data = pdf_storage[username][file_id]
    
    return jsonify({
        "filename": pdf_data["filename"],
        "file_type": pdf_data["file_type"],
        "content": pdf_data["processed_data"]
    })

@app.route("/list_pdfs", methods=["GET"])
def list_pdfs():
    """List all PDFs uploaded by the current user"""
    if not is_logged_in():
        return jsonify({"error": "Not authenticated"}), 401
    
    username = get_username()
    
    if username not in pdf_storage:
        return jsonify({"pdfs": []})
    
    pdfs = []
    for file_id, pdf_data in pdf_storage[username].items():
        pdfs.append({
            "file_id": file_id,
            "filename": pdf_data["filename"],
            "file_type": pdf_data["file_type"],
            "upload_time": pdf_data["upload_time"],
            "summary": pdf_data["processed_data"].get("summary", "")
        })
    
    return jsonify({"pdfs": pdfs})

@app.route("/chat", methods=["POST"])
def chat():
    try:
        # Check if user is logged in
        if not is_logged_in():
            print("Authentication error: User not logged in")
            return jsonify({"reply": "Not authenticated"}), 401
        
        user_id = get_user_id()
        username = get_username()
        user_input = request.json.get("message", "")
        
        # For debugging
        print(f"User input: {user_input}")
        print(f"User ID: {user_id}, Username: {username}")
        
        # Initialize user data if not exists
        if user_id not in message_count:
            message_count[user_id] = {'count': 0, 'timestamp': time.time()}
        
        if user_id not in conversations:
            conversations[user_id] = [{"role": "system", "content": system_prompt}]
        
        # Check if the limit period has passed (reset counter)
        now = time.time()
        if now - message_count[user_id]['timestamp'] > reset_time:
            message_count[user_id] = {'count': 0, 'timestamp': now}

        # Check if the user has exceeded the message limit
        remaining, time_until_reset = get_remaining_messages(user_id)
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
                except Exception as e:
                    print(f"Error in flashcard topic extraction: {str(e)}")
                    topic = user_input  # Fallback to original input
            
            return jsonify({
                "reply": f"I'll create some flashcards about {topic} for you!",
                "flashcard_request": True,
                "topic": topic,
                "remaining_messages": remaining - 1
            })
        
        # Add the user message to conversation history
        conversations[user_id].append({"role": "user", "content": user_input})
        
        # Check if question relates to PDFs
        pdf_keywords = ['pdf', 'document', 'upload', 'file', 'content', 'read', 'understand', 'analyze', 
                       'what does it say', 'paper', 'text', 'in the document']
        pdf_related = any(keyword in user_input.lower() for keyword in pdf_keywords)
        
        # Advanced detection for document references without explicit keywords
        if not pdf_related and user_id in pdf_storage:
            # Check if user is referring to a document without using keywords
            document_reference_phrases = [
                "what does", "tell me about", "summarize", "explain", "analyze",
                "paragraph", "page", "section", "chapter"
            ]
            
            # Look for phrases that might imply referring to a document
            if any(phrase in user_input.lower() for phrase in document_reference_phrases):
                # Get the most recent PDF details
                recent_pdfs = sorted(pdf_storage[user_id].items(), key=lambda x: x[1]['upload_time'], reverse=True)
                if recent_pdfs and (time.time() - recent_pdfs[0][1]['upload_time']) < 300:  # Within 5 minutes
                    # It's likely they're referring to the recently uploaded PDF
                    pdf_related = True
                    print("Detected implicit PDF reference")
        
        # Check if this is a question related to uploaded rubrics
        rubric_keywords = ['rubric', 'criteria', 'grade', 'assessment', 'requirement', 'band', 'score', 'mark']
        is_rubric_question = any(keyword in user_input.lower() for keyword in rubric_keywords)
        
        # Reference PDF content if the question is related
        if (pdf_related or is_rubric_question) and user_id in pdf_storage:
            # Get the most recent PDF uploaded
            if pdf_storage[user_id]:
                recent_pdfs = sorted(pdf_storage[user_id].items(), key=lambda x: x[1]['upload_time'], reverse=True)
                if recent_pdfs:
                    recent_pdf = recent_pdfs[0][1]
                    pdf_type = recent_pdf['file_type']
                    pdf_name = recent_pdf['filename']
                    
                    # Check if this is a direct request for PDF content
                    direct_content_request = any(phrase in user_input.lower() for phrase in 
                                              ["show me the pdf", "what's in the pdf", "content of the pdf",
                                               "what does the document say", "what's in the document"])
                    
                    # Add a reminder about the uploaded PDF
                    pdf_reminder = f"Remember that the user has uploaded a PDF named '{pdf_name}'. "
                    if pdf_type == 'rubric':
                        pdf_reminder += "This is a grading rubric with assessment criteria that should be referenced when answering questions about grades or requirements."
                    else:
                        pdf_reminder += "Please use the content of this document to help answer their question."
                        
                    conversations[user_id].append({"role": "system", "content": pdf_reminder})
                    
                    # If it's a small enough document, include relevant content again
                    processed_data = recent_pdf['processed_data']
                    if processed_data:
                        if pdf_type == 'rubric' and 'tables' in processed_data:
                            rubric_summary = "The rubric contains these grade criteria:\n"
                            for item in processed_data['tables']:
                                rubric_summary += f"- {item['grade']}: {item['description'][:100]}...\n"
                            conversations[user_id].append({"role": "system", "content": rubric_summary})
                        elif 'text' in processed_data:
                            # For direct content requests, always include the content
                            if direct_content_request:
                                full_text = processed_data['text']
                                content_msg = f"Here is the content of '{pdf_name}':\n\n{full_text[:2000]}"
                                if len(full_text) > 2000:
                                    content_msg += f"\n\n[Document continues for {len(full_text) - 2000} more characters...]"
                                conversations[user_id].append({"role": "system", "content": content_msg})
                                print(f"Including full PDF content due to direct request")
                            # For regular requests, include a preview if it's small enough
                            elif len(processed_data['text']) < 2000:
                                conversations[user_id].append({"role": "system", "content": f"Document content (preview): {processed_data['text'][:1000]}..."})
                    
                    print(f"Added PDF context reminder for: {pdf_name}")
        
        # Prepare messages for API, ensuring we stay within token limits
        messages = trim_conversation_history(conversations[user_id])
        
        print(f"Sending request to OpenAI with {len(messages)} messages")
        
        try:
            # Check if client is initialized
            if client is None:
                return jsonify({
                    "reply": "The AI service is not available at the moment. Please check that the OpenAI API key is configured correctly.",
                    "error": "openai_not_configured"
                }), 500
            
            # Try different models in case the preferred one isn't available
            models_to_try = ["gpt-4.1-mini", "gpt-4o-mini", "gpt-3.5-turbo"]
            response = None
            last_error = None
            
            for model in models_to_try:
                try:
                    # Make API call with conversation history
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages
                    )
                    print(f"Successfully used model: {model}")
                    break
                except Exception as e:
                    last_error = e
                    print(f"Failed to use model {model}: {str(e)}")
                    continue
            
            # If all models failed
            if response is None:
                raise last_error or Exception("All models failed")
            
            reply = response.choices[0].message.content
            print(f"Received response from OpenAI: {reply[:100]}...")
            
            # Add assistant's response to conversation history
            conversations[user_id].append({"role": "assistant", "content": reply})
            
            # Increment message count for the user
            message_count[user_id]['count'] += 1
            
            # Also update message count in Firebase
            firebase_auth.update_message_count(user_id)
            
            # Periodically save conversation to Firebase
            if message_count[user_id]['count'] % 5 == 0:  # Save every 5 messages
                firebase_auth.save_conversation(user_id, conversations[user_id])
            
            return jsonify({
                "reply": reply,
                "remaining_messages": remaining - 1
            })

        except Exception as e:
            print(f"OpenAI API Error: {str(e)}")
            # Provide a more specific error message based on the type of exception
            error_message = str(e)
            if "api_key" in error_message.lower():
                return jsonify({"reply": "Sorry, there's an issue with the API key configuration. Please contact support."}), 500
            elif "authentication" in error_message.lower():
                return jsonify({"reply": "Authentication error with the AI service. Please try again later."}), 500
            else:
                return jsonify({"reply": f"Sorry, I encountered an error: {str(e)}"}), 500
                
    except Exception as e:
        # Catch-all for any unexpected errors
        print(f"Unexpected error in chat endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"reply": "Sorry, an unexpected error occurred. Please try again later."}), 500

def trim_conversation_history(messages):
    """Trim conversation history to fit within token limits"""
    # Always keep the system prompt (first message)
    system_message = messages[0]
    
    # Start with just the system message
    trimmed_messages = [system_message]
    current_tokens = num_tokens_from_messages(trimmed_messages)
    
    # Find PDF-related messages first (prioritize keeping these)
    pdf_related_messages = []
    regular_messages = []
    
    for message in messages[1:]:
        content = message.get("content", "").lower()
        role = message.get("role", "")
        
        # Identify PDF-related messages to prioritize
        if role == "system" and any(keyword in content for keyword in ["pdf", "document", "rubric", "grade", "criteria"]):
            pdf_related_messages.append(message)
        elif role == "user" and ("upload" in content and any(keyword in content for keyword in ["pdf", "document", "rubric"])):
            pdf_related_messages.append(message)
        # Check if this is a message containing PDF content
        elif len(content) > 200 and "content of the document" in content:
            # This is likely a PDF content message, keep a shorter version
            shorter_content = content[:500] + "... [PDF content truncated for brevity]"
            pdf_related_messages.append({"role": message["role"], "content": shorter_content})
        else:
            regular_messages.append(message)
    
    # Add the most recent PDF-related messages first
    for message in reversed(pdf_related_messages[-3:]):  # Keep up to 3 most recent PDF messages
        message_tokens = num_tokens_from_messages([message])
        if current_tokens + message_tokens <= MAX_TOKENS:
            trimmed_messages.append(message)
            current_tokens += message_tokens
    
    # Then add as many recent messages as possible, starting from the most recent
    for message in reversed(regular_messages):
        message_tokens = num_tokens_from_messages([message])
        
        if current_tokens + message_tokens <= MAX_TOKENS:
            trimmed_messages.append(message)
            current_tokens += message_tokens
        else:
            # If we can't add any more messages, break
            break
    
    # Sort messages to maintain conversation flow (excluding system prompt)
    conversation_messages = sorted(trimmed_messages[1:], 
                                  key=lambda msg: messages.index(msg) if msg in messages else 999)
    
    return [system_message] + conversation_messages

@app.route("/reset", methods=["POST"])
def reset_conversation():
    # Check if user is logged in
    if not is_logged_in():
        return jsonify({"reply": "Not authenticated"}), 401
    
    user_id = get_user_id()
    
    if user_id in conversations:
        # Keep the system prompt, remove all other messages
        system_message = conversations[user_id][0]
        conversations[user_id] = [system_message]
        
        # Start a new conversation in Firebase
        firebase_auth.save_conversation(user_id, conversations[user_id])
        
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
    if not is_logged_in():
        return jsonify({"error": "Not authenticated"}), 401
    
    try:
        username = get_username()
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

@app.route("/test_openai", methods=["GET"])
def test_openai_connection():
    """Endpoint to test OpenAI API connectivity"""
    try:
        if client is None:
            return jsonify({
                "success": False,
                "error": "OpenAI client not initialized",
                "api_key_set": bool(os.getenv("OPENAI_API_KEY"))
            }), 500
            
        # Try a simple API call
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )
        
        return jsonify({
            "success": True,
            "response": response.choices[0].message.content,
            "model": "gpt-4.1-mini"
        })
        
    except Exception as e:
        error_message = str(e)
        error_type = type(e).__name__
        
        return jsonify({
            "success": False,
            "error": error_message,
            "error_type": error_type,
            "available_models": get_available_models()
        }), 500

def get_available_models():
    """Try to fetch available models"""
    try:
        if client is None:
            return ["Client not initialized"]
        
        models = client.models.list()
        return [model.id for model in models.data]
    except Exception as e:
        return [f"Error listing models: {str(e)}"]

@app.route("/api_status", methods=["GET"])
def api_status():
    """Endpoint to check API connectivity status"""
    status = {
        "app": "NightAtlas AI",
        "status": "running",
        "openai_configured": client is not None,
        "firebase_auth_configured": hasattr(firebase_auth, 'auth') and firebase_auth.auth is not None,
        "firebase_db_configured": hasattr(firebase_auth, 'db') and firebase_auth.db is not None,
        "environment_vars": {
            "openai_api_key": bool(os.getenv("OPENAI_API_KEY")),
            "firebase_api_key": bool(os.getenv("FIREBASE_API_KEY")),
            "firebase_service_account": (
                bool(os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY")) or 
                bool(os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON"))
            )
        }
    }
    return jsonify(status)
    
@app.route("/health", methods=["GET"])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)