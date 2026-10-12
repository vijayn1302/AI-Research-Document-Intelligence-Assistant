import json
from typing import Any, TypedDict

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

from app.config import CHAT_MODEL, GROUNDING_THRESHOLD, MAX_GROUNDING_RETRIES, TOP_K
from app.prompts import SYSTEM_PROMPT, GROUNDING_PROMPT, RELEVANCE_PROMPT
from app.rag import retrieve


class RAGState(TypedDict, total=False):
    question: str
    top_k: int
    context: str
    sources: list[dict[str, Any]]
    answer: str
    grounded: bool
    grounding_score: float
    grounding_reason: str
    relevance_score: float
    relevance_reason: str
    retries: int


llm = ChatGroq(model=CHAT_MODEL, temperature=0)


def ask_llm(prompt: str, **kwargs) -> Any:
    return (ChatPromptTemplate.from_template(prompt) | llm).invoke(kwargs).content


def parse_json(content: Any) -> dict:
    if isinstance(content, dict):
        return content

    if isinstance(content, list):
        content = "".join(
            item if isinstance(item, str) else item.get("text", "")
            for item in content if isinstance(item, (str, dict))
        )

    if not isinstance(content, str):
        return {}

    content = content.strip().removeprefix("```json").removeprefix("```")
    content = content.removesuffix("```").strip()

    try:
        result = json.loads(content)
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        try:
            result = json.loads(content[start:end + 1])
            return result if start >= 0 and end > start and isinstance(result, dict) else {}
        except (json.JSONDecodeError, ValueError):
            return {}


def score(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def retrieve_node(state: RAGState) -> dict:
    pairs = retrieve(state.get("question", ""), state.get("top_k", TOP_K))
    context, sources = [], []

    for i, (doc, similarity) in enumerate(pairs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        page = page + 1 if isinstance(page, int) else page

        context.append(f"[Source {i}: {source}]\n{doc.page_content}")
        sources.append({
            "source": source,
            "page": page,
            "snippet": doc.page_content[:500],
            "score": float(similarity),
        })

    return {"context": "\n\n".join(context), "sources": sources}


def relevance_node(state: RAGState) -> dict:
    if not state.get("context"):
        return {"relevance_score": 0.0, "relevance_reason": "No context retrieved."}

    data = parse_json(ask_llm(
        RELEVANCE_PROMPT,
        question=state.get("question",""),
        context=state.get("context", ""),
    ))

    return {
        "relevance_score": score(data.get("score")),
        "relevance_reason": str(data.get("reason", "Unable to determine relevance.")),
    }


def generate_node(state: RAGState) -> dict:
    answer = ask_llm(
        SYSTEM_PROMPT,
        question=state.get("question", ""),
        context=state.get("context", ""),
    )

    if isinstance(answer, list):
        answer = "".join(
            item if isinstance(item, str) else item.get("text", "")
            for item in answer if isinstance(item, (str, dict))
        )

    return {"answer": str(answer)}



def grounding_node(state: RAGState) -> dict:
    raw_response = ask_llm(
        GROUNDING_PROMPT,
        question=state.get("question", ""),
        context=state.get("context", ""),
        answer=state.get("answer", ""),
    )

    data = parse_json(raw_response)

    grounded_value = data.get("grounded", False)

    if isinstance(grounded_value, bool):
        grounded = grounded_value
    elif isinstance(grounded_value, (int, float)):
        grounded = grounded_value == 1
    elif isinstance(grounded_value, str):
        grounded = grounded_value.strip().lower() in {
            "true", "yes", "1"
        }
    else:
        grounded = False

    grounding_score = score(
        data.get("grounding_score", data.get("score", 0.0))
    )

    grounding_reason = data.get(
        "grounding_reason",
        data.get("reason", "Grounding evaluation returned no reason.")
    )

    if not data or "grounded" not in data:
        print("Invalid grounding response:", repr(raw_response))
        grounded = False
        grounding_score = 0.0
        grounding_reason = "Unable to parse grounding evaluation."

    return {
        "grounded": grounded,
        "grounding_score": grounding_score,
        "grounding_reason": str(grounding_reason),
    }



def retry_node(state: RAGState) -> dict:
    return {
        "retries": state.get("retries", 0) + 1,
        "top_k": state.get("top_k", TOP_K) + 2,
    }


def route(state: RAGState) -> str:
    if state.get("grounded") and state.get("grounding_score", 0) >= GROUNDING_THRESHOLD:
        return "finish"

    return (
        "retry"
        if state.get("retries", 0) < MAX_GROUNDING_RETRIES
        else "finish"
    )


builder = StateGraph(RAGState)

builder.add_node("retrieve", retrieve_node)
builder.add_node("relevance", relevance_node)
builder.add_node("generate", generate_node)
builder.add_node("grounding", grounding_node)
builder.add_node("retry", retry_node)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "relevance")
builder.add_edge("relevance", "generate")
builder.add_edge("generate", "grounding")

builder.add_conditional_edges(
    "grounding", route,
    {"retry": "retry", "finish": END},
)
builder.add_edge("retry", "retrieve")

rag_graph = builder.compile()


def ask(question: str, top_k: int = TOP_K) -> dict:
    question = question.strip()
    if not question:
        raise ValueError("Question is required.")

    return rag_graph.invoke({
        "question": question,
        "top_k": top_k,
        "retries": 0,
    })