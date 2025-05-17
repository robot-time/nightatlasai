import os
import json
import pyrebase
import firebase_admin
from firebase_admin import credentials, firestore, auth as admin_auth
from firebase_config import config, FIREBASE_SERVICE_ACCOUNT_KEY

# Initialize Firebase Admin (for backend operations)
firebase_admin_initialized = False

# Try to initialize Firebase Admin with service account key
try:
    # Option 1: Direct path to service account JSON file
    if os.path.exists(FIREBASE_SERVICE_ACCOUNT_KEY):
        cred = credentials.Certificate(FIREBASE_SERVICE_ACCOUNT_KEY)
        firebase_admin.initialize_app(cred)
        firebase_admin_initialized = True
        print(f"Firebase Admin initialized with service account file: {FIREBASE_SERVICE_ACCOUNT_KEY}")
    
    # Option 2: Service account credentials as JSON string in environment variable
    elif os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON"):
        service_account_info = json.loads(os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON"))
        cred = credentials.Certificate(service_account_info)
        firebase_admin.initialize_app(cred)
        firebase_admin_initialized = True
        print("Firebase Admin initialized with service account JSON from environment variable")
    
    else:
        print("Warning: Firebase service account credentials not found. Admin SDK features will be unavailable.")
        firebase_admin_initialized = False
        
    # Initialize Firestore if admin was initialized
    if firebase_admin_initialized:
        db = firestore.client()
    else:
        db = None
        
except Exception as e:
    print(f"Error initializing Firebase Admin: {str(e)}")
    db = None
    firebase_admin_initialized = False

# Initialize Pyrebase (for authentication)
try:
    firebase = pyrebase.initialize_app(config)
    auth = firebase.auth()
    print("Firebase Authentication initialized successfully")
except Exception as e:
    print(f"Error initializing Firebase Authentication: {str(e)}")
    auth = None

# User Authentication Functions
def register_user(email, password, username):
    """
    Register a new user with email and password
    Returns user data on success, error message on failure
    """
    try:
        # Create the user in Firebase Authentication
        user = auth.create_user_with_email_and_password(email, password)
        
        # If Firebase Admin is initialized, create a custom user profile in Firestore
        if db is not None:
            user_id = user['localId']
            
            # Store additional user info in Firestore
            user_data = {
                'username': username,
                'email': email,
                'created_at': firestore.SERVER_TIMESTAMP,
                'conversations': [],
                'message_count': 0,
                'message_limit': 25,
                'reset_time': 14400,  # 4 hours in seconds
                'last_reset': firestore.SERVER_TIMESTAMP
            }
            
            db.collection('users').document(user_id).set(user_data)
            
            # Use Admin SDK to set custom claims with username
            admin_auth.update_user(
                user_id,
                display_name=username
            )
        
        return {
            'success': True, 
            'user': user,
            'message': 'Registration successful'
        }
    
    except Exception as e:
        # Process Firebase errors
        error_message = str(e)
        if "EMAIL_EXISTS" in error_message:
            return {'success': False, 'message': 'Email already in use'}
        elif "WEAK_PASSWORD" in error_message:
            return {'success': False, 'message': 'Password is too weak'}
        else:
            return {'success': False, 'message': f'Registration failed: {error_message}'}

def login_user(email, password):
    """
    Login a user with email and password
    Returns user data on success, error message on failure
    """
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        
        # Get user profile if Firestore is available
        user_data = {}
        if db is not None:
            user_id = user['localId']
            user_doc = db.collection('users').document(user_id).get()
            if user_doc.exists:
                user_data = user_doc.to_dict()
        
        return {
            'success': True, 
            'user': user,
            'user_data': user_data,
            'message': 'Login successful'
        }
        
    except Exception as e:
        error_message = str(e)
        if "INVALID_PASSWORD" in error_message:
            return {'success': False, 'message': 'Invalid password'}
        elif "EMAIL_NOT_FOUND" in error_message:
            return {'success': False, 'message': 'Email not found'}
        else:
            return {'success': False, 'message': f'Login failed: {error_message}'}

def get_user_data(user_id):
    """
    Get a user's data from Firestore
    """
    if db is None:
        return None
    
    try:
        user_doc = db.collection('users').document(user_id).get()
        if user_doc.exists:
            return user_doc.to_dict()
        return None
    except Exception as e:
        print(f"Error getting user data: {e}")
        return None

def save_conversation(user_id, conversation_data):
    """
    Save a conversation to Firestore
    """
    if db is None:
        return False
    
    try:
        # Create a new conversation document
        conversation_ref = db.collection('users').document(user_id).collection('conversations').document()
        conversation_ref.set({
            'created_at': firestore.SERVER_TIMESTAMP,
            'updated_at': firestore.SERVER_TIMESTAMP,
            'messages': conversation_data
        })
        return True
    except Exception as e:
        print(f"Error saving conversation: {e}")
        return False

def get_conversations(user_id):
    """
    Get all conversations for a user
    """
    if db is None:
        return []
    
    try:
        conversations = []
        conversation_docs = db.collection('users').document(user_id).collection('conversations').order_by('updated_at', direction=firestore.Query.DESCENDING).limit(10).get()
        
        for doc in conversation_docs:
            conversation = doc.to_dict()
            conversation['id'] = doc.id
            conversations.append(conversation)
            
        return conversations
    except Exception as e:
        print(f"Error getting conversations: {e}")
        return []

def save_pdf_data(user_id, pdf_id, pdf_data):
    """
    Save PDF data to Firestore
    """
    if db is None:
        return False
    
    try:
        # Limit the size of processed_data to avoid Firestore document size limitations
        if 'processed_data' in pdf_data and 'text' in pdf_data['processed_data']:
            # Limit text size to prevent Firestore document size issues
            full_text = pdf_data['processed_data']['text']
            if len(full_text) > 10000:  # Firestore has a 1MB document size limit
                pdf_data['processed_data']['text'] = full_text[:10000] + "... [text truncated]"
        
        # Create a new PDF document
        pdf_ref = db.collection('users').document(user_id).collection('pdfs').document(pdf_id)
        pdf_ref.set({
            'created_at': firestore.SERVER_TIMESTAMP,
            **pdf_data
        })
        return True
    except Exception as e:
        print(f"Error saving PDF data: {e}")
        return False

def get_pdfs(user_id):
    """
    Get all PDFs for a user
    """
    if db is None:
        return []
    
    try:
        pdfs = []
        pdf_docs = db.collection('users').document(user_id).collection('pdfs').order_by('created_at', direction=firestore.Query.DESCENDING).get()
        
        for doc in pdf_docs:
            pdf = doc.to_dict()
            pdf['id'] = doc.id
            pdfs.append(pdf)
            
        return pdfs
    except Exception as e:
        print(f"Error getting PDFs: {e}")
        return []

def update_message_count(user_id):
    """
    Update user's message count in Firestore
    """
    if db is None:
        return False
    
    try:
        user_ref = db.collection('users').document(user_id)
        user_doc = user_ref.get()
        
        if user_doc.exists:
            user_data = user_doc.to_dict()
            current_count = user_data.get('message_count', 0)
            user_ref.update({
                'message_count': current_count + 1,
                'last_message_time': firestore.SERVER_TIMESTAMP
            })
        return True
    except Exception as e:
        print(f"Error updating message count: {e}")
        return False 