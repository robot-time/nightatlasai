# NightAtlas AI

NightAtlas AI is an educational AI chat application that supports PDF uploads, flashcards, and grading rubrics.

## Firebase Authentication Setup

This application uses Firebase for authentication and data storage. Follow these steps to set up your Firebase project:

1. **Create a Firebase Project**
   - Go to [Firebase Console](https://console.firebase.google.com/)
   - Click "Add project" and follow the setup wizard
   - Enable Google Analytics if desired

2. **Set Up Authentication**
   - In your Firebase project, go to "Authentication" in the left sidebar
   - Click "Get started"
   - Enable "Email/Password" authentication method

3. **Create Firestore Database**
   - Go to "Firestore Database" in the left sidebar
   - Click "Create database"
   - Choose "Start in production mode" or "Start in test mode" as appropriate
   - Select a location closest to your users

4. **Get Web API Configuration**
   - Go to Project Settings (gear icon near the top left)
   - Scroll down to "Your apps" section
   - Click the web app icon (</>) to register a web app
   - Register the app with a nickname
   - Copy the configuration object that looks like:
   ```javascript
   const firebaseConfig = {
     apiKey: "...",
     authDomain: "...",
     projectId: "...",
     storageBucket: "...",
     messagingSenderId: "...",
     appId: "..."
   };
   ```

5. **Generate Admin SDK Service Account Key**
   - In Project Settings, go to the "Service accounts" tab
   - Click "Generate new private key" button
   - Save the JSON file securely
   - Move it to your project directory (or a secure location)

6. **Update Firebase Configuration**
   - Open `firebase_config.py` and update it with your Web API configuration
   - Set the path to your service account key file

7. **Environment Variables**
   - Create a `.env` file in the project root from the `.env.example` template
   - Add your OpenAI API key and other configuration values

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd nightatlas-ai
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   Open your browser and go to http://127.0.0.1:5000

## Features

- User Authentication with Firebase
- AI Chat with GPT-4.1-mini
- PDF Upload and Analysis
- Special support for grading rubrics
- Flashcard generation
- Conversation history persistence 