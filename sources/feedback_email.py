"""
SMTP email sender for feedback notifications.

Uses Feishu (飞书) enterprise email SMTP by default.
Configuration via environment variables:
  SMTP_HOST    — SMTP server hostname (default: smtp.feishu.com)
  SMTP_PORT    — SMTP server port (default: 465 for SSL)
  SMTP_USER    — SMTP username (full email address)
  SMTP_PASSWORD— SMTP password (Feishu client-specific app password)
  SMTP_FROM    — From address (default: same as SMTP_USER)
  SUPPORT_EMAIL— Support team email to receive feedback (default: support@copiioai.com)
"""

import os
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sources.logger import Logger

logger = Logger("feedback_email.log")


def _get_smtp_config():
    """Read SMTP configuration from environment variables."""
    return {
        "host": os.getenv("SMTP_HOST", "smtp.feishu.com"),
        "port": int(os.getenv("SMTP_PORT", "465")),
        "user": os.getenv("SMTP_USER", ""),
        "password": os.getenv("SMTP_PASSWORD", ""),
        "from_addr": os.getenv("SMTP_FROM", os.getenv("SMTP_USER", "")),
        "support_email": os.getenv("SUPPORT_EMAIL", "support@copiioai.com"),
    }


def send_feedback_notification(user_email: str, user_id: str, feedback_content: str, feedback_id: int) -> bool:
    """
    Send feedback notification email to the support team.

    Args:
        user_email: The email of the user who submitted the feedback
        user_id: The user ID
        feedback_content: The feedback text content
        feedback_id: The feedback record ID in database

    Returns:
        True if sent successfully, False otherwise.
    """
    cfg = _get_smtp_config()

    if not cfg["user"] or not cfg["password"]:
        logger.error("SMTP credentials not configured. Set SMTP_USER and SMTP_PASSWORD in .env")
        return False

    subject = f"[CopiioAI Feedback #{feedback_id}] New feedback from {user_email}"

    body = f"""\
A new feedback has been submitted on CopiioAI.

──────────────────────────────────────
Feedback ID:  #{feedback_id}
User Email:   {user_email}
User ID:      {user_id}
──────────────────────────────────────

Feedback Content:
{feedback_content}

──────────────────────────────────────

To reply to this user, call the admin API:
  POST /admin/send_message
  {{
    "user_id": {user_id},
    "feedback_id": {feedback_id},
    "title": "Your reply title",
    "content": "Your reply content..."
  }}

This will save the message AND email the user automatically.

—
CopiioAI Feedback System
"""

    msg = MIMEMultipart()
    msg["From"] = cfg["from_addr"]
    msg["To"] = cfg["support_email"]
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    try:
        if cfg["port"] == 465:
            # SSL connection
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(cfg["host"], cfg["port"], context=context) as server:
                server.login(cfg["user"], cfg["password"])
                server.sendmail(cfg["from_addr"], cfg["support_email"], msg.as_string())
        else:
            # STARTTLS connection (port 587)
            with smtplib.SMTP(cfg["host"], cfg["port"]) as server:
                server.starttls()
                server.login(cfg["user"], cfg["password"])
                server.sendmail(cfg["from_addr"], cfg["support_email"], msg.as_string())

        logger.info(f"Feedback notification email sent: feedback_id={feedback_id} to={cfg['support_email']}")
        return True

    except smtplib.SMTPAuthenticationError as e:
        logger.error(f"SMTP authentication failed: {e}. Check SMTP_USER and SMTP_PASSWORD.")
        return False
    except smtplib.SMTPException as e:
        logger.error(f"SMTP error sending feedback notification: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error sending email: {e}")
        return False


def send_reply_notification(user_email: str, message_title: str, message_content: str) -> bool:
    """
    Send a reply notification email to the user when admin replies to their feedback.

    Args:
        user_email: The user's email address
        message_title: The message title
        message_content: The message body

    Returns:
        True if sent successfully, False otherwise.
    """
    cfg = _get_smtp_config()

    if not cfg["user"] or not cfg["password"]:
        logger.error("SMTP credentials not configured.")
        return False

    subject = f"[CopiioAI] Reply to your feedback: {message_title}"

    body = f"""\
Hi,

You have received a new reply from the CopiioAI team regarding your feedback.

──────────────────────────────────────
{message_title}
──────────────────────────────────────

{message_content}

──────────────────────────────────────

You can view and manage all messages in the CopiioAI extension popup.

—
CopiioAI Team
"""

    msg = MIMEMultipart()
    msg["From"] = cfg["from_addr"]
    msg["To"] = user_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    try:
        if cfg["port"] == 465:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(cfg["host"], cfg["port"], context=context) as server:
                server.login(cfg["user"], cfg["password"])
                server.sendmail(cfg["from_addr"], user_email, msg.as_string())
        else:
            with smtplib.SMTP(cfg["host"], cfg["port"]) as server:
                server.starttls()
                server.login(cfg["user"], cfg["password"])
                server.sendmail(cfg["from_addr"], user_email, msg.as_string())

        logger.info(f"Reply notification email sent to user: {user_email}")
        return True

    except Exception as e:
        logger.error(f"Error sending reply notification to {user_email}: {e}")
        return False
