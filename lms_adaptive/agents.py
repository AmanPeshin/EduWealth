from typing import TypedDict
from langgraph_supervisor import create_supervisor
from langchain_openai import ChatOpenAI
from langchain.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Chroma
from .config import CHAT_MODEL, OPENAI_API_KEY, TAVILY_API_KEY, VECTOR_DIR

class ChatState(TypedDict):
    messages: list

def build_assistant_graph():
    web = TavilySearchResults(tavily_api_key=TAVILY_API_KEY, max_results=5)
    retriever = Chroma(persist_directory=VECTOR_DIR).as_retriever(search_kwargs={"k": 5})

    research_agent = ChatOpenAI(model=CHAT_MODEL, temperature=0.0, api_key=OPENAI_API_KEY).bind_tools([web])
    tutor_agent = ChatOpenAI(model=CHAT_MODEL, temperature=0.2, api_key=OPENAI_API_KEY)

    def tutor_prompt(messages):
        q = messages[-1]["content"]
        ctx_docs = retriever.invoke(q)
        ctx = "\n\n".join([d.page_content for d in ctx_docs])
        sys = "You are a finance tutor. Prefer the provided context; if insufficient, say you're unsure."
        return [{"role":"system","content":sys},{"role":"user","content":f"Question: {q}\nContext:\n{ctx[:8000]}"}]

    supervisor = create_supervisor(
        model=ChatOpenAI(model=CHAT_MODEL, temperature=0.0, api_key=OPENAI_API_KEY),
        agents=[research_agent, tutor_agent],
        prompt=(
            "Supervisor:\n"
            "- Assign research agent for time-sensitive or 'search' queries.\n"
            "- Assign tutor agent for guidance grounded in company PDFs.\n"
            "Assign exactly one agent per turn."
        ),
        add_handoff_back_messages=True,
        output_mode="full_history",
    ).compile()

    def invoke(state: ChatState, config=None):
        msgs = state["messages"]
        last = msgs[-1]["content"] if msgs else ""
        if any(k in last.lower() for k in ["search", "today", "latest", "market", "news"]):
            return supervisor.invoke({"messages": msgs}, config=config)
        prepared = tutor_prompt(msgs)
        ans = tutor_agent.invoke(prepared).content
        return {"messages": msgs + [{"role":"assistant","content": ans}]}

    return invoke
