import streamlit as st
import os

from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader




# llama-3.1-8b-instant

from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    
    Question: {input}
    """
    
    
)


def create_vectore_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings() 
        st.session_state.loader=PyPDFDirectoryLoader("/research_paper")  ## Data Ingestion
        st.session_state.docs = st.session_state.loader.load() ## Document Laoding
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        
        
prompt = st.text_input("Enter your query from the research paper")
if st.button("Document Embeddings"):
    create_vectore_embedding()
    st.write("Vector Database is ready")
    

import time

if user_prompt:
    document_chain =  create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retriever_chain = create_retriever_chain(retriever, document_chain)
    
    state = time.process_time()
    response = retriever_chain.invoke({'input': user_prompt})
    print(f"Response time:{time.process_time()-start}")
    
    st.write(response['answer'])
    
    ## With a streamlit expander
    with st.expander("Document similarity Search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('------------------')