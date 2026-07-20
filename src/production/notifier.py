import smtplib
from email.mime.text import MIMEText


def send_email(message):

    sender = "tradesig.vm@gmail.com"
    password = "aoky vwai brex ehlw"
    receiver = "maksic.vedran13@gmail.com"

    msg = MIMEText(message, "plain", "utf-8")
    msg["Subject"] = "Daily Trading Signals"
    msg["From"] = sender
    msg["To"] = receiver

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()

    server.login(sender, password)

    server.sendmail(sender, receiver, msg.as_string())

    server.quit()

    print("📧 Email sent!")