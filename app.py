"""
app.py
======
Flask Web Server & REST API for the Merged Phishing Detection System.
Serves the modern Cybersecurity Dashboard frontend and exposes inference endpoints.
"""

import os
import sys
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from merged_phishing_detector import MergedPhishingDetector

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")
CORS(app)

# Global lazy-loaded detector instance
_detector = None


def get_detector():
    global _detector
    if _detector is None:
        print("[*] Initializing MergedPhishingDetector in Flask server...")
        _detector = MergedPhishingDetector()
    return _detector


# Predefined realistic demonstration samples
SAMPLE_CASES = [
    # --- Email Phishing Threats ---
    {
        "id": "phish_paypal",
        "title": "🚨 PayPal Account Suspension",
        "type": "email",
        "subject": "URGENT: Your PayPal Account Has Been Suspended",
        "body": (
            "Dear Customer, We detected unusual login attempts to your account from an unrecognized IP address. "
            "To prevent permanent suspension of your funds, please verify your credentials immediately: "
            "http://secure-paypal-login-verify.xyz/auth/login.php. Thank you, Security Team."
        )
    },
    {
        "id": "phish_m365",
        "title": "🚨 Microsoft 365 Password Expiry",
        "type": "email",
        "subject": "Action Required: Your Microsoft 365 Password Expires in 24 Hours",
        "body": (
            "IT Helpdesk Alert: Your corporate Microsoft 365 password will expire today. "
            "To retain your existing password and prevent email service lockout, please confirm your current credentials "
            "via the IT Portal: http://m365-password-portal-update.xyz/auth/login. Global IT Operations."
        )
    },
    {
        "id": "phish_amazon",
        "title": "🚨 Amazon Unauthorized $1,849 Order",
        "type": "email",
        "subject": "Amazon Order Confirmation - Apple MacBook Pro (#9482-1082)",
        "body": (
            "Thank you for your order! We have charged $1,849.00 to your credit card on file. "
            "If you did NOT authorize this purchase, cancel the unauthorized transaction immediately "
            "to secure your account: http://amazon-dispute-resolution-portal.top/cancel-order.php?id=9482"
        )
    },
    {
        "id": "phish_tax",
        "title": "🚨 IRS Tax Refund Claim Notice",
        "type": "email",
        "subject": "Notice: Pending Tax Refund Confirmation - Action Required",
        "body": (
            "Internal Revenue Service Notification: You have a pending tax refund of $1,420.50. "
            "Submit your identity verification form within 24 hours at "
            "http://irs-tax-refund-portal.top/claim-now.htm or your disbursement will be cancelled."
        )
    },
    {
        "id": "phish_bank",
        "title": "🚨 Bank of America Wire Alert",
        "type": "email",
        "subject": "High Risk Alert: Wire Transfer Pending Authorization ($4,500.00)",
        "body": (
            "Bank of America Fraud Department: An outgoing wire transfer of $4,500.00 to an international beneficiary "
            "is currently on hold. If this transfer was not requested by you, decline the transaction and reset your "
            "Online ID immediately at http://bankofamerica-fraud-verification-team.xyz/security"
        )
    },
    {
        "id": "phish_docusign",
        "title": "🚨 DocuSign Pending Signature",
        "type": "email",
        "subject": "DocuSign: Please DocuSign: Employment Agreement & Tax Form W-4",
        "body": (
            "HR Department has sent you a document for review and electronic signature. "
            "Please review and complete the attached documents before the expiration deadline: "
            "http://docusign-document-review-portal.loan/view-document?token=a89f92. DocuSign Security."
        )
    },
    {
        "id": "phish_crypto",
        "title": "🚨 MetaMask Urgent KYC Verification",
        "type": "email",
        "subject": "MetaMask: Mandatory KYC Regulation Update for Web3 Wallets",
        "body": (
            "Dear User, Under recent decentralized finance compliance regulations, unverified wallets will be restricted. "
            "Verify your 12-word seed recovery phrase and restore full trading permissions: "
            "http://metamask-security-kyc-validation.site/verify-seed-phrase. Web3 Support."
        )
    },

    # --- Legitimate / Safe Emails ---
    {
        "id": "legit_newsletter",
        "title": "🛡️ Python Weekly Tech Newsletter",
        "type": "email",
        "subject": "Python Weekly Newsletter - Issue #520",
        "body": (
            "Hello everyone, Welcome to this week's issue of Python Weekly. In this issue: "
            "Read about the new features in Python 3.13 at https://python.org/whatsnew and explore "
            "open source tutorials on GitHub at https://github.com/trending. Happy coding!"
        )
    },
    {
        "id": "legit_github",
        "title": "🛡️ GitHub PR Review Notification",
        "type": "email",
        "subject": "[GitHub] Pull Request #142 approved on tensorflow/keras",
        "body": (
            "Hi Vibhu, Your pull request #142 'Add support for custom metrics callbacks' has been approved "
            "by the maintainers. View the full diff and discussion on GitHub: "
            "https://github.com/keras-team/keras/pull/142. Thanks for your contribution!"
        )
    },
    {
        "id": "legit_meeting",
        "title": "🛡️ Internal Business Roadmap Sync",
        "type": "email",
        "subject": "Quarterly Planning & Team Sync Meeting Notes",
        "body": (
            "Hi team, Attached are the action items from our roadmap discussion today. "
            "Please review the deck before Friday's standup. Let me know if you need any adjustments "
            "to the timeline. Best regards, Alex."
        )
    },
    {
        "id": "legit_stripe",
        "title": "🛡️ Stripe Subscription Confirmation",
        "type": "email",
        "subject": "Stripe: Your Monthly Developer Subscription Has Renewed",
        "body": (
            "Hi Vibhu, Thank you for being a customer. Your cloud developer subscription has successfully renewed "
            "for the upcoming billing cycle. You can view your account settings and manage your services anytime "
            "at https://billing.stripe.com/p/session/live_109284019284. Best regards, Stripe Team."
        )
    },
    {
        "id": "legit_university",
        "title": "🛡️ Academic Research Collaboration",
        "type": "email",
        "subject": "Draft Paper: Deep Learning Applications in Cybersecurity",
        "body": (
            "Dear Professor, I have finished updating the experimental section of our upcoming conference manuscript. "
            "The model comparison table now includes the latest benchmark numbers from our test evaluation runs. "
            "Looking forward to discussing the results during our weekly meeting on Thursday."
        )
    },

    # --- URL Phishing Threats ---
    {
        "id": "url_phish_ip",
        "title": "🚨 Raw IP Host with Open Port",
        "type": "url",
        "url": "http://192.168.1.105:8080/secure/bank-login-verify/update.php?id=99281&token=auth"
    },
    {
        "id": "url_phish_paypal",
        "title": "🚨 PayPal Spoof on Suspicious .xyz TLD",
        "type": "url",
        "url": "http://paypal-account-security-update.xyz/verify/login.php"
    },
    {
        "id": "url_phish_apple",
        "title": "🚨 Apple ID Subdomain Typosquatting (.top)",
        "type": "url",
        "url": "http://appleid.apple.com.secure-verify-session-id9910.top/auth/login"
    },
    {
        "id": "url_phish_at",
        "title": "🚨 Credential Injection URL with @ Symbol",
        "type": "url",
        "url": "http://support.google.com@malicious-harvesting-site.ru/login.php"
    },

    # --- Legitimate / Safe URLs ---
    {
        "id": "url_legit_wiki",
        "title": "🛡️ Wikipedia Phishing Article",
        "type": "url",
        "url": "https://en.wikipedia.org/wiki/Phishing"
    },
    {
        "id": "url_legit_google",
        "title": "🛡️ Google Developers Machine Learning",
        "type": "url",
        "url": "https://developers.google.com/machine-learning/crash-course"
    },
    {
        "id": "url_legit_python",
        "title": "🛡️ Official Python 3 Documentation",
        "type": "url",
        "url": "https://www.python.org/downloads/release/python-3120/"
    },
    {
        "id": "url_legit_github",
        "title": "🛡️ GitHub Official Linux Repository",
        "type": "url",
        "url": "https://github.com/torvalds/linux"
    }
]


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "online", "model": "Merged Phishing Detector (DistilBERT + URL DNN)"})


@app.route("/api/samples", methods=["GET"])
def get_samples():
    return jsonify(SAMPLE_CASES)


@app.route("/api/predict/email", methods=["POST"])
def predict_email():
    try:
        data = request.get_json(force=True) or {}
        subject = data.get("subject", "")
        body = data.get("body", "")

        if not subject and not body:
            return jsonify({"error": "Please provide an email subject or body."}), 400

        detector = get_detector()
        result = detector.predict(subject=subject, body=body)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict/url", methods=["POST"])
def predict_url():
    try:
        data = request.get_json(force=True) or {}
        url = data.get("url", "").strip()

        if not url:
            return jsonify({"error": "Please provide a valid URL."}), 400

        detector = get_detector()
        result = detector.predict(url=url)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/benchmarks", methods=["GET"])
def get_benchmarks():
    return jsonify({
        "models": [
            {
                "name": "URL Deep Neural Network (DNN)",
                "scope": "Lexical & Structural URL Features (49 features)",
                "samples": 126511,
                "accuracy": 96.23,
                "precision": 95.35,
                "recall": 96.18,
                "f1": 0.9576,
                "roc_auc": 0.9918,
                "framework": "TensorFlow / Keras 3"
            },
            {
                "name": "Email DistilBERT Transformer",
                "scope": "Email Subject & Body NLP Intent",
                "samples": 8395,
                "accuracy": 99.33,
                "precision": 99.55,
                "recall": 99.13,
                "f1": 0.9934,
                "roc_auc": 0.9998,
                "framework": "PyTorch / HuggingFace"
            },
            {
                "name": "Merged Multimodal Dual-Engine",
                "scope": "Full Email NLP + Embedded Link Deep Extraction",
                "samples": 83948,
                "accuracy": 98.82,
                "precision": 98.40,
                "recall": 100.00,
                "f1": 0.9919,
                "roc_auc": 0.9882,
                "framework": "DistilBERT + URL DNN Fusion"
            }
        ],
        "hardware": "NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM) + Intel Core",
        "url_coverage_in_emails": "50.7% of emails contain embedded links"
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n[🚀] Starting Phishing Detection Web Server on http://127.0.0.1:{port}")
    # Preload detector on startup
    get_detector()
    app.run(host="127.0.0.1", port=port, debug=False)
