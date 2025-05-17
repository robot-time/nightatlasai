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
    
    # Option 2: Service account credentials as JSON string in FIREBASE_SERVICE_ACCOUNT_JSON
    elif os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON"):
        service_account_info = json.loads(os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON"))
        cred = credentials.Certificate(service_account_info)
        firebase_admin.initialize_app(cred)
        firebase_admin_initialized = True
        print("Firebase Admin initialized with service account JSON from FIREBASE_SERVICE_ACCOUNT_JSON")
    
    # Option 3: Service account credentials as JSON string in FIREBASE_SERVICE_ACCOUNT_KEY
    elif FIREBASE_SERVICE_ACCOUNT_KEY and FIREBASE_SERVICE_ACCOUNT_KEY.strip().startswith('{'):
        try:
            # Try to parse the JSON directly
            try:
                service_account_info = json.loads(FIREBASE_SERVICE_ACCOUNT_KEY)
            except json.JSONDecodeError:
                # If direct parsing fails, try to clean up the string
                # Sometimes environment variables can have escape characters
                cleaned_json = FIREBASE_SERVICE_ACCOUNT_KEY.replace('\\"', '"').replace("\\n", "\n")
                service_account_info = json.loads(cleaned_json)
                
            cred = credentials.Certificate(service_account_info)
            firebase_admin.initialize_app(cred)
            firebase_admin_initialized = True
            print("Firebase Admin initialized with service account JSON from FIREBASE_SERVICE_ACCOUNT_KEY")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: FIREBASE_SERVICE_ACCOUNT_KEY appears to contain JSON but could not be parsed: {str(e)}")
            # Fallback: Try to write the content to a temporary file
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
                    temp_file.write(FIREBASE_SERVICE_ACCOUNT_KEY.encode('utf-8'))
                    temp_file_path = temp_file.name
                
                print(f"Wrote service account key to temporary file: {temp_file_path}")
                cred = credentials.Certificate(temp_file_path)
                firebase_admin.initialize_app(cred)
                firebase_admin_initialized = True
                print(f"Firebase Admin initialized with temporary service account file")
                
                # Clean up the temporary file after initialization
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
            except Exception as e2:
                print(f"Failed to create temporary file for service account key: {str(e2)}")
                firebase_admin_initialized = False
    
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

# Add Google sign-in function
def sign_in_with_google_token(id_token):
    """
    Authenticate a user with a Google ID token
    Returns user data on success, error message on failure
    """
    try:
        # Verify the ID token with Firebase
        # Try to use sign_in_with_id_token if available, otherwise fall back to custom token
        try:
            # First try to use the ID token directly
            user = auth.sign_in_with_id_token(id_token)
        except AttributeError:
            # If sign_in_with_id_token is not available in the Pyrebase version, 
            # we need to verify the token with Firebase Admin and then sign in
            if firebase_admin_initialized:
                # Verify the Google ID token using Firebase Admin SDK
                decoded_token = admin_auth.verify_id_token(id_token)
                uid = decoded_token['uid']
                
                # Create a custom token for this user
                custom_token = admin_auth.create_custom_token(uid)
                
                # Sign in with the custom token
                user = auth.sign_in_with_custom_token(custom_token)
            else:
                raise Exception("Firebase Admin SDK not initialized, cannot verify Google token")
        
        # Get user profile if Firestore is available
        user_data = {}
        if db is not None:
            user_id = user['localId']
            user_doc = db.collection('users').document(user_id).get()
            
            # If user document doesn't exist yet, create it
            if not user_doc.exists:
                # Extract email and display name from token info
                account_info = auth.get_account_info(user['idToken'])
                email = account_info['users'][0]['email']
                display_name = account_info['users'][0].get('displayName', email.split('@')[0])
                
                # Create new user document in Firestore
                user_data = {
                    'username': display_name,
                    'email': email,
                    'created_at': firestore.SERVER_TIMESTAMP,
                    'conversations': [],
                    'message_count': 0,
                    'message_limit': 25,
                    'reset_time': 14400,  # 4 hours in seconds
                    'last_reset': firestore.SERVER_TIMESTAMP
                }
                
                db.collection('users').document(user_id).set(user_data)
            else:
                user_data = user_doc.to_dict()
        
        return {
            'success': True, 
            'user': user,
            'user_data': user_data,
            'message': 'Login successful'
        }
        
    except Exception as e:
        error_message = str(e)
        return {'success': False, 'message': f'Google login failed: {error_message}'} 