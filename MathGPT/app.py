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
    description="Useful only for general knowledge questions about people, places, history, or science. Not for math calculations."
)


## Initiliza the Math tool
math_chain=LLMMathChain.from_llm(llm=llm)
calculator=Tool(
    name="Calculator",
    func=math_chain.run,
    description="Useful for solving math problems, arithmetic calculations, and word problems involving numbers."
)


prompt = """
You are a reasoning assistant.

Solve the following problem step by step and explain clearly.

Question: {question}

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
def generate_response(question):
    response = assistant_agent.invoke({"input": question})
    return response

## Lets start the interaction
question = st.text_area("Enter your question:", "A box has 9 red balls and 11 blue balls. If 3 red balls and 5 blue balls are taken out, how many balls remain in the box?")

if st.button("Find my answer"):
    if question:
        with st.spinner("Generating response..."):
            
            # Save user message
            st.session_state.messages.append({
                "role": "user",
                "content": question
            })

            st.chat_message("user").write(question)

            # LangChain callback
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

            # Run agent
            response = assistant_agent.run(question, callbacks=[st_cb])

            # Save assistant response
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })

            st.chat_message("assistant").write(response)

    else:
        st.warning("Please enter the question...")