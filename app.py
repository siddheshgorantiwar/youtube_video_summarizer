import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

## Streamlit APP Configuration
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", 
                   page_icon="🦜", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        /* General Text Styling */
        body, .stTextInput input, .stButton button, .stMarkdown, .stAlert, .stException, .title {
            color: #333333 !important;
        }
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f7f8fc;
        }
        .stApp {
            background-color: #f0f2f6;
        }
        .title {
            text-align: center;
            font-size: 3rem;
            font-weight: bold;
            color: #333333 !important; /* Ensure the title color is applied */
        }
        .subtitle {
            text-align: center;
            font-size: 1.5rem;
            color: #555555;
            margin-bottom: 1rem;
        }
        .stButton button {
            border-radius: 12px;
            background-color: #0066cc;
            color: white;
            font-size: 1.2rem;
            padding: 10px 20px;
        }
        .stButton button:hover {
            background-color: #005bb5;
        }
        .sidebar .sidebar-content {
            background-color: #dde1e7;
            padding: 2rem;
            border-radius: 10px;
        }
        .stTextInput input {
            border-radius: 10px;
            border: 2px solid #0066cc;
            padding: 10px;
            font-size: 1.1rem;
            color: #333333;
            background-color: #ffffff;
        }
        .stTextInput input::placeholder {
            color: #808080;
        }
        .stAlert p {
            font-size: 1.1rem;
            color: #333333 !important;
        }
    </style>
    """, unsafe_allow_html=True)

# Title and Subtitle with Emojis
st.markdown('<div class="title">🦜 LangChain: Summarize Text From YT or Website</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Summarize URL</div>', unsafe_allow_html=True)

## Sidebar: Get the Groq API Key and URL (YT or website) to be summarized
with st.sidebar:
    st.markdown("## 🔐 Enter your Groq API Key")
    groq_api_key = st.text_input("Groq API Key", value="", type="password")
    st.markdown("## 🌐 Enter the URL (YouTube or Website)")
    generic_url = st.text_input("URL", label_visibility="collapsed", placeholder="https://example.com")

## Gemma Model Using Groq API
llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

## Summarize Button and Action
if st.button("✨ Summarize the Content from YT or Website"):
    # Validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("🚨 Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("🚨 Please enter a valid URL. It can be a YouTube video URL or website URL")
    else:
        try:
            with st.spinner("⏳ Summarizing content..."):
                # Loading the website or YouTube video data
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url], 
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                    )
                docs = loader.load()

                # Chain for Summarization
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                st.success("✅ Summary Generated Successfully!")
                st.markdown(f"### 📄 Summary:\n{output_summary}")
        except Exception as e:
            st.exception(f"⚠️ Exception: {e}")
