# embeddings_llm.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
import torch
from langchain_huggingface import HuggingFaceEmbeddings

def create_vectorstore(chunks, embedding_model="sentence-transformers/all-mpnet-base-v2"):
    """Create Chroma vectorstore from document chunks."""
    model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
    encode_kwargs = {"normalize_embeddings": False}
    hf = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    return Chroma.from_documents(documents=chunks, embedding=hf)

def initialize_agent(tools, llm_model="gemini-2.0-flash-001"):
    """Initialize Gemini agent (no .env required)."""
    GEMINI_API_KEY = "AIzaSyCA-lxDO324cyulMI3MSx-2APsYJpom26Y"  # ⚠️ Replace with your actual Gemini key
    llm = ChatGoogleGenerativeAI(
        model=llm_model,
        temperature=0.3,
        google_api_key=GEMINI_API_KEY
    )
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

def run_query(agent_executor, query: str):
    """Run query through LangChain agent."""
    return agent_executor.invoke({"input": query})
