import streamlit as st
import requests

# Set page configuration
st.set_page_config(
    page_title="Claim Matching Demo",
    page_icon="✅",
    layout="centered",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    /* General App Styling */
    body {
        font-family: "Segoe UI", Roboto, sans-serif;
        background-color: #F9F9F9;
    }
    .stButton>button {
        background-color: #4CAF50 !important;
        color: white !important;
        border-radius: 4px !important;
    }
    .stTextInput>div>div>input {
        border-radius: 4px !important;
        border: 1px solid #ccc !important;
    }
    .stSelectbox>div>div>div>select {
        border-radius: 4px !important;
    }
    .stRadio>div>div>label {
        font-size: 0.95rem !important;
    }
    .title {
        text-align: center;
        margin-top: 0px;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-top: 0px;
        margin-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and subtitle
st.markdown("<h1 class='title'>Claim Matching Application</h1>", unsafe_allow_html=True)
st.markdown("<h4 class='subtitle'>Identify Similar or Relevant Claims Quickly & Easily</h4>", unsafe_allow_html=True)

# Brief description / instructions
st.write(
    """
    This tool helps you determine whether a new claim is similar or related to existing claims 
    in your knowledge base. Simply select your mode of operation, enter the claim to verify, 
    choose how many similar claims you’d like to retrieve, and click **Search**. 
    """
)

# Create a horizontal divider
st.markdown("---")

# Mode selection (sidebar or main — here, we'll keep it in main for clarity)
mode = st.selectbox(
    "Mode Selection:",
    ["OpenAI", "Local"],
    help="Choose whether to use OpenAI's API or a local model."
)

# Number of similar claims to display
num_similar = st.number_input(
    "Number of similar claims to display:",
    min_value=1,
    max_value=10,
    value=3,
    step=1,
    help="How many similar/relevant claims to show in the results."
)

# Claim input
claim = st.text_area(
    "Enter a claim to verify:",
    "",
    help="Type or paste the claim statement you wish to analyze."
)

# Additional options for OpenAI mode
if mode == "OpenAI":
    openai_key = st.text_input(
        "OpenAI API Key:",
        type="password",
        help="Your personal or organizational OpenAI API key."
    )
    model_choice = st.radio(
        "OpenAI Model:",
        ["gpt-3.5-turbo-0125", "gpt-4o", "gpt-4-turbo"],
        help="Select the desired model for claim matching."
    )

# Search button
if st.button("Search"):
    if not claim.strip():
        st.warning("Please enter a valid claim before searching.")
    else:
        try:
            if mode == "Local":
                # Send a POST request to your local endpoint
                response = requests.post(
                    "http://localhost:8000/classify",
                    json={"claim": claim}
                )
                results = response.json()
            else:
                # Ensure we have an API key for OpenAI mode
                if not openai_key:
                    st.error("OpenAI API key is required for OpenAI mode.")
                    st.stop()
                # Send a POST request to the OpenAI-based endpoint
                response = requests.post(
                    "http://localhost:8000/classifyopenai",
                    json={
                        "claim": claim,
                        "model": model_choice,
                        "api_key": openai_key
                    }
                )
                results = response.json()

            # Display results
            st.markdown("### Top-K Similar Claims:")
            # Safely handle case where "top_k" may not exist in the JSON
            top_k = results.get("top_k", [])
            if not top_k:
                st.info("No similar claims were found.")
            else:
                for idx, (similar_claim, score) in enumerate(top_k[:num_similar], start=1):
                    st.write(f"**{idx}.** {similar_claim} \nScore: {score*100:.0f}%")
        
        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred while connecting to the server: {e}")
        except ValueError:
            st.error("Received invalid JSON response from the server.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

st.markdown("---")
