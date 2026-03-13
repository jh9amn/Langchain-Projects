import validators, streamlit as st
from langchain_classic.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader


## Streamlit APP
st.set_page_config(page_title="LangChain: Text Summarization from YT or Website", page_icon="🧠")
st.title("LangChain: Text Summarization from YT or Website")
st.subheader("Enter a YouTube video URL or a website URL to get a summary of the content.")

## Get the Groq API key and url(YT or website) to be summarized
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed", placeholder="Enter a YouTube video URL or a website URL")
 
# Prompt template for summarization
prompt_template = """
Provide a summary of the following content in 300 words or less:

Content:
{text}
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["text"]
)

if st.button("Summarize the Content from YT or Website"):
    disabled=not groq_api_key or not generic_url
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information in the input fields.")

    elif not validators.url(generic_url):
        st.error("Please enter a valid URL.")

    else:
        try:
            with st.spinner("Waiting..."):

                llm = ChatGroq(
                    model="llama-3.1-8b-instant",   # or "llama3-70b-8192" llama-3.1-8b-instant
                    temperature=0.7,
                    api_key=groq_api_key
                )

                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(
                        generic_url,
                        add_video_info=True
                    )
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": "Mozilla/5.0"
                        }
                    )

                data = loader.load()

                chain = load_summarize_chain(
                    llm,
                    chain_type = "stuff" if len(data) < 5 else "map_reduce",
                    prompt=prompt,
                    verbose=False,
                    token_max=3000
                )

                output_summary = chain.run(data)
                st.success(output_summary)

        except Exception as e:
            st.exception(e)