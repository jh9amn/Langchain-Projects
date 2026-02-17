import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchResults
from langchain_classic.agents import initialize_agent, AgentType
from langchain_classic.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

## Arxiv and wikipedia tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchResults(name="Search")

st.title("Langchain - Chat with search")

"""
In this example, we're using "StreamlitCallbackHandler" to display the thought and actions of an agent in an interactive Streamlit app.
Try more LangChain ❤️ Streamlit Agent examples at [github.com/hwchase17/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent)
"""

## Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Groq API Key", type="password")

if "message" not in st.session_state:
    st.session_state["message"] = [
        {
            "role": "assistant",
            "content": "Hello! I'm an agent that can search the web and query arxiv and wikipedia. How can I help you today?",
        }
    ]

for msg in st.session_state["message"]:
    st.chat_message(msg["role"]).write(msg["content"])
    
if prompt := st.chat_input(placeholder="What is Machine Learning?"):
    st.session_state.message.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    llm = ChatGroq(groq_api_key=api_key, 
                model = "llama-3.1-8b-instant",
                streaming=True)
    
    tools = [arxiv, wiki, search]
    
    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handling_parsing_errors=True
    )
    
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.message, callbacks=[st_cb])
        st.session_state.message.append({"role": "assistant", "content": response})
        st.write(response)