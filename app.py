# app.py
import streamlit as st
import tempfile
import pandas as pd
from loaders import load_pdf, split_documents
from embeddings_llm import create_vectorstore, initialize_agent, run_query
from stock_tools import fetch_stock_data, compute_indicators, stock_summary
from langchain.tools import Tool
from langchain.tools.retriever import create_retriever_tool

st.set_page_config(page_title="📊 Stock Report AI Assistant", layout="wide")
st.title("🤖 AI Stock Report & Market Assistant")

# Sidebar Instructions
with st.sidebar.expander("ℹ️ Instructions", expanded=True):
    st.markdown("""
    **Capabilities:**
    - Upload company stock reports (PDFs)
    - Ask financial questions (e.g., revenue growth, profit trends)
    - Get live market data (RSI, MA20, MA50)
    - Combines PDF + live data insights
    """)

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "stock_data" not in st.session_state:
    st.session_state.stock_data = None

# --- Upload Stock Reports (PDFs) ---
uploaded_files = st.file_uploader("📂 Upload Stock Report PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files and st.session_state.vectorstore is None:
    st.info("⏳ Processing stock reports...")
    all_chunks = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        docs = load_pdf(tmp_path)
        chunks = split_documents(docs)
        all_chunks.extend(chunks)

    st.session_state.vectorstore = create_vectorstore(all_chunks)
    st.session_state.retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 8})
    st.success("✅ Stock reports processed successfully!")

# --- Live Stock Data ---
st.sidebar.subheader("📈 Live Stock Data")
ticker = st.sidebar.text_input("Enter Stock Symbol (e.g. AAPL, TCS.NS):", "AAPL")
if st.sidebar.button("Fetch Live Data"):
    with st.spinner(f"Fetching {ticker} data..."):
        data = fetch_stock_data(ticker)
        data = compute_indicators(data)

        # ✅ FIX: Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

        st.session_state.stock_data = data
        st.success(f"✅ Fetched {ticker} data successfully!")

# --- Display Stock Chart ---
if st.session_state.stock_data is not None:
    try:
        st.line_chart(st.session_state.stock_data[["Close", "MA20", "MA50"]])
    except KeyError:
        st.warning("⚠️ Could not plot all indicators — some columns might be missing.")
    st.markdown(stock_summary(st.session_state.stock_data, ticker))

# --- Build Agent ---
tools = []
if st.session_state.retriever:
    tools.append(create_retriever_tool(
        st.session_state.retriever,
        name="stock_report_search",
        description="Search company stock report PDFs for financial details."
    ))

if st.session_state.stock_data is not None:
    def stock_data_tool(query: str):
        summary = stock_summary(st.session_state.stock_data, ticker)
        return f"{summary}\nNow analyze based on this: {query}"
    tools.append(Tool(
        name="live_stock_data_tool",
        func=stock_data_tool,
        description="Analyzes live stock indicators like RSI, MA20, and MA50."
    ))

if tools and st.session_state.agent is None:
    st.session_state.agent = initialize_agent(tools)

# --- Chat Interface ---
st.subheader("💬 Chat with AI (Reports + Market Data)")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask about financial performance or stock trend...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    if st.session_state.agent is None:
        with st.chat_message("assistant"):
            st.warning("⚠️ Please upload stock PDFs or fetch data first.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("🤖 Analyzing..."):
                try:
                    result = run_query(st.session_state.agent, query)
                    answer = result["output"]
                except Exception as e:
                    answer = f"❌ Error: {str(e)}"
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
