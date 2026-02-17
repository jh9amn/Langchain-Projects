# 🔍 LangChain Search Agent with Streamlit & Groq

An interactive **AI-powered chat application** built using **Streamlit**, **LangChain**, and **Groq LLMs**.  
This app allows users to ask questions and get answers by **searching the web**, **querying Wikipedia**, and **fetching research papers from arXiv** — all in real time.

---

## 🚀 Features

- 💬 Chat-based UI using **Streamlit**
- 🔎 Web search via **DuckDuckGo**
- 📚 Knowledge retrieval from **Wikipedia**
- 🧪 Research paper search using **arXiv**
- ⚡ Fast inference using **Groq LLMs**
- 🧠 Agent reasoning with visible thought/action steps
- 🔄 Streaming responses for better UX

---

## 🧠 How It Works (High-Level Flow)

1. User enters a query in the chat interface  
2. A **LangChain Agent** decides:
   - Should I search the web?
   - Should I query Wikipedia?
   - Should I fetch an arXiv paper?
3. The agent calls the appropriate tool(s)
4. Results are summarized by the **Groq-powered LLM**
5. The response is streamed live to the UI

---

## 🛠️ Tech Stack

- **Streamlit** → Frontend UI
- **LangChain (Classic)** → Agent orchestration
- **Groq (LLaMA 3.1)** → Large Language Model
- **DuckDuckGo** → Web search
- **Wikipedia API** → General knowledge
- **arXiv API** → Research papers

---

## 🧩 Tools Used

### 🔹 arXiv Tool
### 🔹 Wikipedia Tool
### 🔹 DuckDuckGo Search Tool



-----------------------------
## 🏗️ Application Architecture

```text
User (Chat Input)
        │
        ▼
 Streamlit UI (Chat Interface)
        │
        ▼
 LangChain Agent (Zero-Shot ReAct)
        │
        ├── DuckDuckGo Search Tool
        ├── Wikipedia Query Tool
        └── arXiv Research Tool
        │
        ▼
 Groq LLM (LLaMA 3.1 – Streaming)
        │
        ▼
 Final Answer (Streamed to UI)

