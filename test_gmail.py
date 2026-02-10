import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

# Load .env file if it exists
if os.path.exists('.env'):
    with open('.env') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

# --- Configuration ---
# You can set these in your environment, or edit them here for testing
GMAIL_USER = 'sahooaiagent@gmail.com'
GMAIL_PASSWORD = os.environ.get('GMAIL_APP_PASSWORD', '') 
RECEIVER_EMAIL = 'sahooaiagent@gmail.com'

def test_gmail_connection():
    if not GMAIL_PASSWORD:
        print("Error: GMAIL_APP_PASSWORD not found in environment.")
        password = input("Please enter your 16-character Gmail App Password for testing: ").strip()
    else:
        password = GMAIL_PASSWORD

    print(f"Testing Gmail connection for {GMAIL_USER}...")
    
    subject = "AMA Pro Scanner: Test Connection"
    body = "This is a test email from your AMA Pro Scanner script. If you received this, your Gmail connection is working correctly!"

    message = MIMEMultipart()
    message["From"] = GMAIL_USER
    message["To"] = RECEIVER_EMAIL
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(GMAIL_USER, password)
            server.sendmail(GMAIL_USER, RECEIVER_EMAIL, message.as_string())
        print("\nSUCCESS: Test email sent successfully!")
    except Exception as e:
        print(f"\nFAILURE: Could not send test email. Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure 2-Step Verification is enabled on your Google Account.")
        print("2. Ensure you are using an 'App Password', not your regular Gmail password.")
        print("3. Check your internet connection.")

if __name__ == "__main__":
    test_gmail_connection()
