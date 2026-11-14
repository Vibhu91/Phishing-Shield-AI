"""
quick_check.py
==============
Simple, clean script to test any Email or URL against your trained models.

Usage:
    phishing_env\\Scripts\\python.exe quick_check.py
"""

from merged_phishing_detector import MergedPhishingDetector

def main():
    print("=" * 65)
    print("       PHISHING SHIELD AI - INSTANT THREAT SCANNER")
    print("=" * 65)
    
    # 1. Initialize detector (loads models onto GPU)
    detector = MergedPhishingDetector()
    
    # 2. Check a URL
    test_url = "http://secure-paypal-login-verify.xyz/auth/login.php"
    print(f"\n[1] Scanning URL: {test_url}")
    url_res = detector.predict_url(test_url)
    print(f"    Verdict : {url_res['verdict']} [{url_res['risk_level']}]")
    print(f"    Prob    : {url_res['phishing_prob'] * 100:.2f}%")
    print(f"    Signals : {', '.join(url_res['key_signals'])}")
    
    # 3. Check an Email Body
    test_email = (
        "URGENT: Your account access has been restricted due to unauthorized login attempts. "
        "Please confirm your credentials immediately to avoid permanent suspension."
    )
    print(f"\n[2] Scanning Email Text:")
    print(f"    Text    : \"{test_email[:70]}...\"")
    email_res = detector.predict_email_text(subject="Account Alert", body=test_email)
    print(f"    Verdict : {email_res['verdict']}")
    print(f"    Prob    : {email_res['phishing_prob'] * 100:.2f}%")

    # 4. Full Multimodal Fusion (Email text + embedded URL)
    print(f"\n[3] Full Pipeline (Email Text + Embedded URL):")
    full_res = detector.predict(
        subject="Action Required", 
        body=f"{test_email} Verify here: {test_url}"
    )
    print(f"    Verdict     : {full_res['verdict']} [{full_res['risk_level']}]")
    print(f"    Threat Prob : {full_res['combined_phishing_prob'] * 100:.2f}%")
    print(f"    Explanation : {full_res['explanation']}")
    print("\n" + "=" * 65)

if __name__ == "__main__":
    main()
