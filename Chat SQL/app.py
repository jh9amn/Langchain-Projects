## https://docs.langchain.com/oss/python/langchain/sql-agent

import streamlit as st
from pathlib import Path
# from langchain.agents import create_sql_agent
from langchain_community.agent_toolkits.sql.base import create_sql_agent
# from langchain.sql_database import SQLDatabase
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_classic.agents import AgentType
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_classic.callbacks import StreamlitCallbackHandler
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq


st.set_page_config(page_title="LangChain SQL Agent", page_icon="🐦")
st.title("🐦 LangChain SQL Agent")

INJECTION_WARNING = """
SQL agent can be vulnerable to prompt injection. Use a DB role with limited
"""

LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

radio_opt = ["Use SQLLite 3 Database. Students.db", "Connect to my MySQL Database"]

selected_opt = st.sidebar.radio(label="Select Database Option", options = radio_opt)


if radio_opt.index(selected_opt) == 1:
    db_url = MYSQL
    mysql_host = st.sidebar.text_input("Provide MySQL Host")
    mysql_user = st.sidebar.text_input("Provide MySQL User")
    mysql_password = st.sidebar.text_input("Provide MySQL Password", type="password")
    mysql_db = st.sidebar.text_input("Provide MySQL Database Name")
else:
    db_url = LOCALDB
    
api_key = st.sidebar.text_input("Provide Groq API Key", type="password")

if not db_url:
    st.info("Please enter the database connection details to proceed.")
    
if not api_key:
    st.info("Please enter the Groq API key to proceed.")
    st.stop()
    
## Call LLM model
llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant", streaming=True)

# @st.cache_resource(ttl=2 * 60 * 60)  # 2 hours
@st.cache_resource(ttl="2h")
def configure_db(db_uri, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None):
    try:
        if db_uri == LOCALDB:
            dbfilepath = (Path(__file__).parent / "students.db").resolve()

            if not dbfilepath.exists():
                st.error(f"Database not found at: {dbfilepath}")
                st.stop()

            engine = create_engine(f"sqlite:///{dbfilepath}")
            return SQLDatabase(engine)

        elif db_uri == MYSQL:
            if not all([mysql_host, mysql_user, mysql_password, mysql_db]):
                st.error("Please provide all MySQL connection details.")
                st.stop()

            engine = create_engine(
                f"mysql+mysqlconnector://{mysql_user}:{mysql_password}"
                f"@{mysql_host}/{mysql_db}"
            )
            return SQLDatabase(engine)
    except Exception as e:
        st.error(f"Connection Error: {e}")
        st.stop()
        
if db_url == MYSQL:
    db=configure_db(db_url, mysql_host, mysql_user, mysql_password, mysql_db)
else:
    db = configure_db(db_url)
    
    
## toolkit to use SQL database
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True  # Added to handle LLM format errors
)

if "messages" not in st.session_state or st.sidebar.button("Clear Conversation"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I assist you with your database?"}]
    
## Apending Messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    

user_query = st.chat_input(placeholder="Ask a anything from the database...")

if user_query:
    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())

        response = agent.invoke(
            {"input": user_query},
            callbacks=[streamlit_callback]
        )

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )
        st.write(response)
    
    