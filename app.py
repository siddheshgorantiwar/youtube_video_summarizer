import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

## Streamlit APP Configuration
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", 
                   page_icon="🦜", layout="wide")

# Title and Subtitle
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL")

## Sidebar: Get the Groq API Key and URL (YT or website) to be summarized
with st.sidebar:
    st.header("🔐 Enter your Groq API Key")
    groq_api_key = st.text_input("Groq API Key", value="", type="password")
    st.header("🌐 Enter the URL (YouTube or Website)")
    generic_url = st.text_input("URL", placeholder="https://example.com")

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

                if not docs:
                    st.error("🚨 Could not retrieve content from the URL. Please check if the URL is correct.")
                else:
                    # Chain for Summarization
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    
                    # Use invoke instead of run
                    output_summary = chain.invoke({"input_documents": docs})
                    
                    st.success("✅ Summary Generated Successfully!")
                    st.markdown(f"### 📄 Summary:\n{output_summary['output_text']}")
        except Exception as e:
            st.exception(f"⚠️ Exception: {e}")
