import os


class Settings:
    app_name: str = "WebCraft Store"
    secret_key: str = os.environ.get("SECRET_KEY", "change-me-in-production-use-a-long-random-string")

    # PayPal — replace with your sandbox or live Client ID from
    # https://developer.paypal.com/dashboard/applications
    paypal_client_id: str = os.environ.get("PAYPAL_CLIENT_ID", "YOUR_PAYPAL_CLIENT_ID_HERE")
    # "sandbox" or "production"
    paypal_env: str = os.environ.get("PAYPAL_ENV", "sandbox")

    # PayPal.me username (e.g. "yourname" → paypal.me/yourname)
    paypal_me_username: str = os.environ.get("PAYPAL_ME_USERNAME", "")
    # Venmo @handle (without the @)
    venmo_username: str = os.environ.get("VENMO_USERNAME", "")

    # ── Contact form email ────────────────────────────────────────────────────
    # The address that receives contact form submissions
    contact_recipient: str = os.environ.get("CONTACT_RECIPIENT", "rt75272@gmail.com")
    # SMTP server settings (defaults work for Gmail)
    smtp_host: str = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port: int = int(os.environ.get("SMTP_PORT", 587))
    # Gmail address used to SEND the email
    smtp_user: str = os.environ.get("SMTP_USER", "")
    # Gmail App Password (NOT your regular password).
    # Generate one at: https://myaccount.google.com/apppasswords
    smtp_password: str = os.environ.get("SMTP_PASSWORD", "")

settings = Settings()
