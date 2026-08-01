import streamlit as st
import pandas as pd
import yfinance as yf
import warnings

# 1. SILENCE VERSION WARNINGS (For Python 3.14 Compatibility)
warnings.filterwarnings("ignore", category=UserWarning, module='pydantic')

# 2. 2026 MODULAR IMPORTS
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain.memory import ConversationBufferMemory

# ==========================================
# PAGE SETUP
# ==========================================
st.set_page_config(page_title="AI Financial Assistant", page_icon="💼", layout="wide")
st.title("💼 Personalized AI Financial Assistant")
st.markdown("---")

# ==========================================
# DATA LOADING (Fixed Filename)
# ==========================================
@st.cache_data
def load_data():
    try:
        return pd.read_csv("synthetic_personal_finance_dataset.csv")
    except Exception as e:
        st.error(f"CSV File Not Found! Ensure it is in the same folder as app.py")
        return pd.DataFrame()

df = load_data()

# ==========================================
# TOOLS (ReAct Logic)
# ==========================================
# Move tool implementations into tools.py to make them testable and reusable
from tools import load_data, get_stock_price, get_client_profile

# reload df (tools.load_data returns normalized columns)
df = load_data()

# Build LangChain Tool objects only if langchain-core import is available
try:
    from langchain_core.tools import Tool as LC_Tool
    tools = [
        LC_Tool(name="GetLiveStockPrice", func=get_stock_price, description="Get recent stock price. Input: ticker symbol (e.g. AAPL)."),
        LC_Tool(name="GetClientProfile", func=lambda uid: get_client_profile(df, uid), description="Get client profile from dataset. Input: User ID (e.g. U00001)")
    ]
except Exception:
    # Fallback shape for tools when langchain not available
    tools = [
        {"name": "GetLiveStockPrice", "func": get_stock_price},
        {"name": "GetClientProfile", "func": (lambda uid: get_client_profile(df, uid))}
    ]

# Small helper: Mock LLM for offline testing
import os
USE_MOCK_LLM = os.environ.get('USE_MOCK_LLM', 'false').lower() in ('1', 'true', 'yes')

class MockAssistant:
    def __init__(self):
        pass
    def respond(self, user_input: str):
        # Very small heuristic responder that uses tools
        u = user_input.lower()
        if 'savings' in u or 'save' in u:
            # try to extract user id like U00001
            words = user_input.split()
            uid = next((w for w in words if w.upper().startswith('U') and w[1:].isdigit()), None)
            if uid:
                profile = get_client_profile(df, uid)
                if 'message' in profile:
                    return profile['message']
                return f"User {uid} has savings: {profile.get('savings_usd', 'N/A')} and monthly income {profile.get('monthly_income_usd', 'N/A')}"
            return "Please specify a client ID like U00001 to fetch savings."
        if 'price' in u or 'stock' in u:
            # extract ticker (last token)
            words = [w.strip(',.') for w in user_input.split()]
            ticker = words[-1].upper()
            return get_stock_price(ticker)
        return "I am a mock assistant. For full responses run with Ollama (set USE_MOCK_LLM=false)."

mock_assistant = MockAssistant() if USE_MOCK_LLM else None

# ==========================================
# BRAIN SETUP (Ollama + LangChain)
# ==========================================
# Try to wire up the LLM/agent. If Ollama or LangChain pieces fail, fall back to mock assistant for testing.
try:
    llm = ChatOllama(model="my-finance-bot", temperature=0)

    # The ReAct prompt template (Required for create_react_agent in 2026)
    template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

History: {chat_history}
Question: {input}
Thought: {agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template)

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Build the Agent and Executor
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        memory=st.session_state.memory, 
        verbose=True, 
        handle_parsing_errors=True
    )
except Exception as e:
    agent_executor = None
    # keep going with mock assistant if available
    st.warning(f"LLM/agent not available ({e}). Running in degraded/mock mode. Set USE_MOCK_LLM=1 for a deterministic mock assistant.)")

# ==========================================
# USER INTERFACE
# ==========================================
# Sidebar: choose client and mode
st.sidebar.header("Session")
client_ids = []
if not df.empty:
    # try to find user id column
    uid_cols = [c for c in df.columns if 'user' in c and 'id' in c or c == 'user_id']
    if not uid_cols:
        uid_cols = [df.columns[0]]
    client_ids = sorted(df[uid_cols[0]].astype(str).unique().tolist())

selected_client = st.sidebar.selectbox("Select client ID", options=["(none)"] + client_ids)
use_mock = st.sidebar.checkbox("Use mock assistant (no Ollama)", value=USE_MOCK_LLM)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "I am connected to your custom Llama-3 brain. How can I help with your finance data today?"}]

# Show client profile if selected
if selected_client and selected_client != "(none)":
    profile = get_client_profile(df, selected_client)
    st.subheader(f"Client {selected_client}")
    if 'message' in profile:
        st.info(profile['message'])
    else:
        # pretty display
        st.write(profile)
        # simple charts: spending vs income vs savings
        try:
            income = float(profile.get('monthly_income_usd', 0) or 0)
            expenses = float(profile.get('monthly_expenses_usd', 0) or 0)
            savings = float(profile.get('savings_usd', 0) or 0)
            st.subheader("Quick financial snapshot")
            st.bar_chart({"Income vs Expenses vs Savings": [income, expenses, savings]})
        except Exception:
            pass

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("Ask me about client U00001 or ask for a stock price...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Brain is thinking..."):
            # Decide how to get a response
            if use_mock or agent_executor is None:
                # use mock assistant
                response = mock_assistant.respond(user_input) if mock_assistant else "Mock assistant not enabled."
            else:
                try:
                    result = agent_executor.invoke({"input": user_input})
                    response = result.get("output") or str(result)
                except Exception as e:
                    response = f"Agent failed: {e}"
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
