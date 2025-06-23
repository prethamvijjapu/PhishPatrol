import streamlit as st
import pandas as pd
import joblib
from urllib.parse import urlparse

# Load the trained model
model = joblib.load('voting_classifier.pkl')

def extract_features(url):
    parsed = urlparse(url)
    features = {
        'having_IPhaving_IP_Address': 1 if any(char.isdigit() for char in parsed.netloc.replace('.', '')) else -1,
        'URLURL_Length': 1 if len(url) > 75 else -1,
        'Shortining_Service': 1 if any(s in parsed.netloc for s in ['bit.ly', 'goo.gl', 'tinyurl', 'shorte.st']) else -1,
        'having_At_Symbol': 1 if '@' in url else -1,
        'double_slash_redirecting': 1 if '//' in parsed.path else -1,
        'Prefix_Suffix': 1 if '-' in parsed.netloc else -1,
        'having_Sub_Domain': 1 if parsed.netloc.count('.') > 2 else -1,
        'SSLfinal_State': 1 if parsed.scheme == 'https' else -1,
        'Domain_registeration_length': -1,
        'Favicon': -1,
        'port': 1 if parsed.port not in [80, 443, None] else -1,
        'HTTPS_token': 1 if 'https' in parsed.netloc else -1,
        'Request_URL': -1,
        'URL_of_Anchor': -1,
        'Links_in_tags': -1,
        'SFH': -1,
        'Submitting_to_email': -1,
        'Abnormal_URL': 1 if len(url.split('/')) > 6 else -1,
        'Redirect': -1,
        'on_mouseover': -1,
        'RightClick': -1,
        'popUpWidnow': 1 if 'popup' in parsed.path else -1,
        'Iframe': -1,
        'age_of_domain': -1,
        'DNSRecord': -1,
        'web_traffic': -1,
        'Page_Rank': -1,
        'Google_Index': -1,
        'Links_pointing_to_page': -1,
        'Statistical_report': -1
    }
    return features

st.title("🔒 PhishPatrol")
url = st.text_input("Enter URL to analyze:", "https://example.com")

if st.button("Analyze"):
    try:
        features = extract_features(url)
        input_df = pd.DataFrame([features])
        
        suspicious_features = [k for k, v in features.items() if v == 1]
        
        if len(suspicious_features) > 0:
            st.error("⚠️ Phishing Website Detected")
            st.write("Suspicious Features:")
            st.write(suspicious_features)
        else:
            st.success("✅ Legitimate Website")
        
        st.write("### Feature Details:")
        for feature, value in features.items():
            if value == 1:
                st.write(f"- {feature}: Suspicious")
            elif value == -1:
                st.write(f"- {feature}: Safe")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")

st.sidebar.info("This tool uses a decision tree model to detect potential phishing websites based on URL characteristics.")
