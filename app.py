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
def get_stock_price(ticker: str) -> str:
    try:
        stock = yf.Ticker(ticker)
        price = stock.info.get('currentPrice', 'N/A')
        return f"The current live price of {ticker} is ${price}."
    except:
        return f"Could not fetch price for {ticker}."

def get_client_profile(user_id: str) -> str:
    if df.empty: return "Database is empty."
    # Matches the first column of your Kaggle CSV
    client = df[df.iloc[:, 0] == user_id] 
    if client.empty: return f"Client {user_id} not found."
    return f"Client Data: {client.iloc[0].to_dict()}"

tools = [
    Tool(name="GetLiveStockPrice", func=get_stock_price, description="Use to get current stock prices. Input: Ticker symbol (e.g. AAPL)"),
    Tool(name="GetClientProfile", func=get_client_profile, description="Use to get client financial data. Input: User ID (e.g. U00001)")
]

# ==========================================
# BRAIN SETUP (Ollama + LangChain)
# ==========================================
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

# ==========================================
# USER INTERFACE
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "I am connected to your custom Llama-3 brain. How can I help with your finance data today?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_input := st.chat_input("Ask me about client U00001..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Brain is thinking..."):
            # Use .invoke() instead of .run() (2026 standard)
            result = agent_executor.invoke({"input": user_input})
            response = result["output"]
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})