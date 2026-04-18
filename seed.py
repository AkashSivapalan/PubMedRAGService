"""
seed.py — Script for initializing the chromadb with chunked PubMed articles or various topics. 
"""

from ingest import ingest_all_topics, MEDICAL_TOPICS

if __name__ == "__main__":
    print(f"Seeding knowledge base with {len(MEDICAL_TOPICS)} medical topics.")
    print("Topics to ingest:")
    for i, (query, max_results) in enumerate(MEDICAL_TOPICS, 1):
        print(f"  {i:2}. {query} ({max_results} articles)")
    print()

    chunks = ingest_all_topics()
    print(f"\nSeeding complete. {len(chunks)} total chunks written to ./chroma_db")
    print("You can now start the server with: uvicorn main:app --reload")
