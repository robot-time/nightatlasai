# Deploying NightAtlas AI to Render

This guide walks you through deploying the NightAtlas AI application to [Render](https://render.com/).

## Prerequisites

- A [Render](https://render.com/) account
- Your Firebase project set up with authentication and Firestore
- Your Firebase service account key (JSON file)
- OpenAI API key

## Deployment Steps

1. **Create a New Web Service on Render**

   - Log in to your Render account
   - Click "New +" and select "Web Service"
   - Connect your GitHub repository
   - Give your service a name (e.g., "nightatlas-ai")
   - Set the runtime to "Python 3"
   - Set the build command: `pip install -r requirements.txt`
   - Set the start command: `gunicorn app:app`
   - Click "Create Web Service"

2. **Add Environment Variables**

   Navigate to the "Environment" tab in your Render dashboard and add the following environment variables:

   ```
   FIREBASE_API_KEY=your_api_key
   FIREBASE_AUTH_DOMAIN=nightatlas-ai.firebaseapp.com
   FIREBASE_PROJECT_ID=nightatlas-ai
   FIREBASE_STORAGE_BUCKET=nightatlas-ai.firebasestorage.app
   FIREBASE_MESSAGING_SENDER_ID=68778559010
   FIREBASE_APP_ID=1:68778559010:web:879dc0af5e46f44cf0ac7b
   FIREBASE_MEASUREMENT_ID=G-JKTB6XV71V
   FIREBASE_DATABASE_URL=https://nightatlas-ai.firebaseio.com
   OPENAI_API_KEY=your_openai_api_key
   SECRET_KEY=generate_a_secure_random_key
   ```

3. **Add Service Account Credentials**

   You have two options for adding Firebase service account credentials:

   **Option 1: Add as an environment variable**
   
   - Open your `serviceAccountKey.json` file
   - Copy the ENTIRE content of the file
   - Add it as an environment variable on Render:
     ```
     FIREBASE_SERVICE_ACCOUNT_JSON={"type":"service_account","project_id":"...","private_key_id":"...","private_key":"...","client_email":"...","client_id":"...","auth_uri":"...","token_uri":"...","auth_provider_x509_cert_url":"...","client_x509_cert_url":"..."}
     ```
   
   **Option 2: Add as a secret file**
   
   - In Render, go to the "Secrets" tab
   - Click "Add Secret File"
   - Set the file name to `serviceAccountKey.json`
   - Paste the content of your service account key file
   - Set the mount path to `/etc/secrets/firebase/`
   - Add `FIREBASE_SERVICE_ACCOUNT_KEY=/etc/secrets/firebase/serviceAccountKey.json` to your environment variables

4. **Add gunicorn to requirements**

   Make sure `gunicorn` is included in your `requirements.txt` file:
   ```
   flask
   openai>=1.0.0
   python-dotenv
   tiktoken
   werkzeug
   PyPDF2
   firebase-admin
   pyrebase4
   gunicorn
   ```

5. **Configure Persistent Disk for Uploads**

   - In the Render dashboard, go to the "Disks" tab
   - Click "Create Disk"
   - Set a name for your disk (e.g., "nightatlas-uploads")
   - Set the mount path to `/app/uploads`
   - Choose the size based on your needs (start with 1 GB)

6. **Deploy Your Application**

   - Go back to the "Overview" tab
   - Click "Manual Deploy" and select "Deploy latest commit"
   - Wait for the deployment to complete

## Troubleshooting

- **Service account issues**: Check your service account JSON format and ensure there are no extra spaces or newlines.
- **Authentication errors**: Verify your Firebase API key and credentials.
- **OpenAI API errors**: Make sure your OpenAI API key is valid and has credits available.
- **Application errors**: Check the Render logs for specific error messages.

## Updating Your Application

To update your application:

1. Push changes to your GitHub repository
2. Render will automatically deploy the latest changes (if auto-deploy is enabled)
3. Or manually deploy from the Render dashboard 