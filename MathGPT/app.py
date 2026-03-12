import streamlit as st
from langchain_groq import ChatGroq
from langchain_classic.chains.llm_math.base import LLMMathChain
from langchain_classic.chains.llm import LLMChain
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_classic.agents import Tool, initialize_agent, AgentType
from langchain_classic.callbacks import StreamlitCallbackHandler
from langchain_core.prompts import PromptTemplate


## Set the Streamlit APP
st.set_page_config(
    page_title="Text to Math Problem Solver and Data Search Assistant",
    page_icon="🧮"
)
st.title("Text to math problem solver using Open Sources")

groq_api_key = st.sidebar.text_input(label="Groq API key", type="password")

if not groq_api_key:
    st.info("Please add your Groq API key to continue..")
    st.stop()
    
llm  = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key, temperature=0
)

## Intializing the tools
wikipedia_wrapper = WikipediaAPIWrapper()

wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the Internet to find the vatious information on the topics mentioned"
)


## Initiliza the Math tool
math_chain=LLMMathChain.from_llm(llm=llm)
calculator=Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tools for answering math related questions. Only input mathematical expression need to bed provided"
)

prompt="""
Your a agent tasked for solving users mathemtical question. Logically arrive at the solution and provide a detailed explanation
and display it point wise for the question below
Question:{question}
Answer:
"""

prompt_template = PromptTemplate(
    input_variables = ["question"],
    template=prompt
)

## Combine all the tools into chain
chain = LLMChain(llm=llm, prompt=prompt_template)

reasoning_tool = Tool(
    name="Reasoing Tool",
    func=chain.run,
    description="A tool for answering logic-based and reasoing question.."
)

## initialize the agents
assistant_agent = initialize_agent(
    tools = [wikipedia_tool, calculator, reasoning_tool],
    llm = llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,   
    verbose=True,
    handle_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state['messages'] = [
        {"role": "assistant", "content": "Hi, I'm math chatbot who can answer all your math problems."}
    ]
    
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])
    

## function to generate the response
# def generate_response(question):
#     response = assistant_agent.invoke({"input": question})
#     return response

## Lets start the interaction
question=st.text_area("Enter youe question:","I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?")

if st.button("find my answer"):
    if question:
        with st.spinner("Generate response.."):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)

            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

            response = assistant_agent.run(question, callbacks=[st_cb])

            st.session_state.messages.append({"role":"assistant","content":response})
            st.write("### Response:")
            st.success(response)

    else:
        st.warning("Please enter the question")






