from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import cohere
import os
from dotenv import load_dotenv

load_dotenv()

co = cohere.Client(os.getenv("COHERE_API_KEY"))


class HybridRetriever:
    def __init__(self, chunks: list[Document], persist_dir: str = "./chroma_db"):
        self.chunks = chunks

        # Load existing ChromaDB (already populated by ingest())
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name="pubmed"
        )

        # BM25 index over same chunks
        tokenized = [doc.page_content.lower().split() for doc in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def vector_search(self, query: str, k: int = 20) -> list[Document]:
        return self.vectorstore.similarity_search(query, k=k)

    def bm25_search(self, query: str, k: int = 20) -> list[Document]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.chunks[i] for i in top_indices]

    def rerank(self, query: str, documents: list[Document], top_n: int = 5) -> list[Document]:
        if not documents:
            return []

        results = co.rerank(
            query=query,
            documents=[doc.page_content for doc in documents],
            top_n=top_n,
            model="rerank-english-v3.0"
        )
        return [documents[r.index] for r in results.results]

    def retrieve(self, query: str, top_n: int = 5) -> list[Document]:
        # Stage 1: candidates from both retrievers
        vector_results = self.vector_search(query, k=20)
        bm25_results = self.bm25_search(query, k=20)

        # Merge and deduplicate by content
        seen = set()
        combined = []
        for doc in vector_results + bm25_results:
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                combined.append(doc)

        # Stage 2: rerank merged candidates
        return self.rerank(query, combined, top_n=top_n)
