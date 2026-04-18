import requests
import xml.etree.ElementTree as ET
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import time
from dotenv import load_dotenv

load_dotenv()

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
CHROMA_PATH = "./chroma_db"

#Medical topics to extract from PubMed to chunk and store in db
MEDICAL_TOPICS = [
    ("cardiology heart failure treatment",              200),
    ("heart failure ejection fraction outcomes",        200),
    ("cardiac resynchronization therapy heart failure", 150),
    ("heart failure hospitalization mortality",         150),
    ("HFpEF HFrEF treatment guidelines",               150),
    ("heart failure biomarkers BNP troponin",           100),
    ("heart failure pharmacotherapy ACE inhibitors",    150),
    ("heart failure device therapy ICD",                100),
]


def search_pubmed(query: str, max_results: int = 200) -> list[str]:
    """Search PubMed and return list of PMIDs."""
    resp = requests.get(f"{EUTILS_BASE}/esearch.fcgi", params={
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json"
    })
    return resp.json()["esearchresult"]["idlist"]


def fetch_abstracts(pmids: list[str]) -> list[Document]:
    """Fetch full abstracts for a list of PMIDs."""
    documents = []
    for i in range(0, len(pmids), 50):
        batch = pmids[i:i+50]
        resp = requests.get(f"{EUTILS_BASE}/efetch.fcgi", params={
            "db": "pubmed",
            "id": ",".join(batch),
            "rettype": "abstract",
            "retmode": "xml"
        })
        root = ET.fromstring(resp.content)
        for article in root.findall(".//PubmedArticle"):
            pmid    = article.findtext(".//PMID", "")
            title   = article.findtext(".//ArticleTitle", "")
            abstract = article.findtext(".//AbstractText", "")
            journal = article.findtext(".//Journal/Title", "")
            year    = article.findtext(".//PubDate/Year", "")
            authors = [
                f"{a.findtext('LastName', '')} {a.findtext('ForeName', '')}".strip()
                for a in article.findall(".//Author")[:3]
            ]

            if not abstract:
                continue

            content = (
                f"Title: {title}\n"
                f"Authors: {', '.join(authors)}\n"
                f"Journal: {journal} ({year})\n\n"
                f"Abstract: {abstract}"
            )
            documents.append(Document(
                page_content=content,
                metadata={
                    "pmid": pmid,
                    "title": title,
                    "journal": journal,
                    "year": year,
                    "source": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                }
            ))
        time.sleep(0.34)  # PubMed rate limit: 3 req/sec
    return documents


def ingest(
    query: str,
    max_results: int = 200,
    replace: bool = True
) -> list[Document]:
    """
    Fetch from PubMed, chunk, embed, and persist to ChromaDB.

    Args:
        query:       PubMed search query
        max_results: max articles to fetch
        replace:     True  = wipe existing collection first (single topic)
                     False = append to existing collection (multi-topic)
    """
    print(f"  Searching PubMed: '{query}'")
    pmids = search_pubmed(query, max_results)
    print(f"  Found {len(pmids)} articles. Fetching abstracts...")

    documents = fetch_abstracts(pmids)
    print(f"  Fetched {len(documents)} abstracts with content.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=75,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"  Created {len(chunks)} chunks.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if replace:
        # Wipe and rebuild the collection from scratch
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PATH,
            collection_name="pubmed"
        )
    else:
        # Append to the existing collection
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings,
            collection_name="pubmed"
        )
        vectorstore.add_documents(chunks)

    print(f"  Persisted to ChromaDB.")
    return chunks


def ingest_all_topics(topics: list[tuple[str, int]] = None) -> list[Document]:
    """
    Ingest a list of (query, max_results) topic tuples into one ChromaDB collection.
    The first topic replaces any existing collection; subsequent topics append.

    Args:
        topics: list of (query, max_results) tuples.
                Defaults to MEDICAL_TOPICS if not provided.
    """
    if topics is None:
        topics = MEDICAL_TOPICS

    all_chunks = []
    total = len(topics)

    for i, (query, max_results) in enumerate(topics):
        print(f"\n[{i+1}/{total}] Ingesting: '{query}'")
        replace = (i == 0)  # only wipe on the first topic
        chunks = ingest(query, max_results, replace=replace)
        all_chunks.extend(chunks)

    print(f"\nDone. Total chunks across all topics: {len(all_chunks)}")
    return all_chunks
