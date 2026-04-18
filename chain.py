from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from retriever import HybridRetriever

SYSTEM_PROMPT = """You are a cardiology research assistant with expertise in heart failure treatment and outcomes.

Answer the user's question using ONLY the provided research context. Always:
- Cite specific studies by title or journal when making claims
- Acknowledge when evidence is limited or conflicting  
- Distinguish between guideline-recommended treatments and emerging evidence
- Reference specific drug classes, devices, or biomarkers when relevant
- Never hallucinate studies or statistics not present in the context

Research Context:
{context}

Conversation History:
{history}

Question: {question}

Answer (with citations):"""


class PubMedRAGChain:
    def __init__(self, retriever: HybridRetriever, model: str = "gpt-4o"):
        self.retriever = retriever
        self.llm = ChatOpenAI(model=model, temperature=0.1)
        self.prompt = PromptTemplate(
            input_variables=["context", "history", "question"],
            template=SYSTEM_PROMPT
        )

    def _format_history(self, history: list[tuple[str, str]]) -> str:
        """Render history tuples as plain text for the prompt."""
        if not history:
            return "No previous conversation."
        lines = []
        for question, answer in history:
            lines.append(f"User: {question}")
            lines.append(f"Assistant: {answer}")
        return "\n".join(lines)

    def format_context(self, docs: list[Document]) -> str:
        sections = []
        for i, doc in enumerate(docs):
            sections.append(
                f"[Source {i+1}] {doc.metadata.get('title', 'Unknown')}\n"
                f"Journal: {doc.metadata.get('journal', 'Unknown')} ({doc.metadata.get('year', '')})\n"
                f"URL: {doc.metadata.get('source', '')}\n"
                f"{doc.page_content}\n"
            )
        return "\n---\n".join(sections)

    def query(self, question: str, history: list[tuple[str, str]]) -> dict:
        """
        Run a RAG query. History is passed in from the caller (main.py)
        and managed externally in Redis — this class is stateless.
        """
        docs = self.retriever.retrieve(question, top_n=5)
        context = self.format_context(docs)
        history_text = self._format_history(history)

        prompt_text = self.prompt.format(
            context=context,
            history=history_text,
            question=question
        )
        response = self.llm.invoke(prompt_text)
        return {"answer": response.content, "sources": docs}