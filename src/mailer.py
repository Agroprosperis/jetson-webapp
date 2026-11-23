import os
import re
import smtplib

from email.message import EmailMessage


EMAIL_ADDRESS_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
SMTP_TIMEOUT = 30


def send_email(settings, recipients, subject, body, attachments):
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = settings["sender"]
    message["To"] = ", ".join(recipients)
    message.set_content(body)

    for filename, payload in attachments:
        message.add_attachment(
            payload,
            maintype="image",
            subtype=os.path.splitext(filename)[1].lstrip(".").lower() or "jpeg",
            filename=filename,
        )

    host = settings["host"]
    port = settings["port"]
    tls_mode = settings["tls_mode"]
    try:
        if tls_mode == "ssl":
            server = smtplib.SMTP_SSL(host, port, timeout=SMTP_TIMEOUT)
        else:
            server = smtplib.SMTP(host, port, timeout=SMTP_TIMEOUT)
        with server:
            if tls_mode == "starttls":
                server.starttls()
            if settings["username"]:
                server.login(settings["username"], settings["password"])
            server.send_message(message)
    except (smtplib.SMTPException, OSError) as exc:
        raise RuntimeError(f"Не вдалося надіслати лист: {exc}") from exc
